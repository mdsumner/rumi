#!/usr/bin/env python3
"""
rumi_bench.py - extended decode benchmark: GDAL (GTiff / LIBERTIFF) vs tifffile vs rumi

Extends asterisk-labs/rumi examples/rumi-vs-geotiff.ipynb with:
  * GTiff re-encodings of the same pixels (codec x interleave x block size, no
    overviews) so codec and layout effects can be separated
  * extra engines: tifffile (+ imagecodecs if present), LIBERTIFF when available
  * GDAL config sweeps: thread mechanism, GDAL_CACHEMAX, dataset reuse (cache
    trap), preallocated output, pixel- vs band-interleaved output, /vsimem vs
    disk, GTIFF_DIRECT_IO / GTIFF_VIRTUAL_MEM_IO for uncompressed
  * rumi cases: recipes, tile size, checksum verification on/off, bytes vs path
  * windowed and single-band reads for every engine that supports them
  * rumi runs each (threads, verify) combination in a fresh subprocess, because
    rumi pins its thread pool and checksum setting at the first read

Uses the system GDAL via osgeo.gdal (no rasterio: its wheel bundles a second
libgdal). Every output is normalised to (band, row, col) and hashed; rows that
do not match the reference pixels are flagged, not silently dropped.

Setup in the gdal-r-ci extras image (keeps numpy at the version the GDAL
bindings were built against):

    python -c "import numpy; print('numpy==' + numpy.__version__)" > /tmp/np.txt
    uv pip install -c /tmp/np.txt "rumi-eo[write]" tifffile imagecodecs

Examples:

    python rumi_bench.py                                # default S2 TCI scene
    python rumi_bench.py --threads 1,4,16,all --reps 7
    python rumi_bench.py --src my.tif --variants orig,zstd-band-1024
    python rumi_bench.py --sections env,full,rumi --csv out.csv
    python rumi_bench.py --config GDAL_NUM_THREADS=ALL_CPUS --config FOO=BAR
    python rumi_bench.py --profile                      # geozl recipe survey
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time
import urllib.request

import numpy as np

DEFAULT_URL = ("https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
               "37/M/BV/2024/10/S2A_37MBV_20241029_0_L2A/TCI.tif")

# name -> GTiff creation options; "{bs}" is filled from the name suffix.
# Needs codec support in the GDAL build; unavailable ones are skipped.
VARIANTS = {
    "deflate-pixel-1024": ["COMPRESS=DEFLATE", "PREDICTOR=2", "INTERLEAVE=PIXEL"],
    "deflate-band-1024":  ["COMPRESS=DEFLATE", "PREDICTOR=2", "INTERLEAVE=BAND"],
    "deflate-band-512":   ["COMPRESS=DEFLATE", "PREDICTOR=2", "INTERLEAVE=BAND"],
    "zstd-pixel-1024":    ["COMPRESS=ZSTD", "PREDICTOR=2", "INTERLEAVE=PIXEL"],
    "zstd-band-1024":     ["COMPRESS=ZSTD", "PREDICTOR=2", "INTERLEAVE=BAND"],
    "lzw-band-1024":      ["COMPRESS=LZW", "PREDICTOR=2", "INTERLEAVE=BAND"],
    "lerczstd-band-1024": ["COMPRESS=LERC_ZSTD", "MAX_Z_ERROR=0", "INTERLEAVE=BAND"],
    "webp-pixel-1024":    ["COMPRESS=WEBP", "WEBP_LOSSLESS=TRUE", "INTERLEAVE=PIXEL"],
    "jxl-band-1024":      ["COMPRESS=JXL", "JXL_LOSSLESS=YES", "JXL_EFFORT=3",
                           "INTERLEAVE=BAND"],
    "none-band-1024":     ["COMPRESS=NONE", "INTERLEAVE=BAND"],
    "none-pixel-1024":    ["COMPRESS=NONE", "INTERLEAVE=PIXEL"],
}
DEFAULT_VARIANTS = ["orig", "deflate-pixel-1024", "deflate-band-1024", "zstd-band-1024",
                    "lerczstd-band-1024", "none-band-1024"]

# rumi frame layouts (what one compressed frame holds, per tile):
#   bhw  all bands in one frame, band-sequential inside it (the notebook's)
#   b    one frame per band per tile (like TIFF PlanarConfiguration=2)
#   hwb  all bands in one frame, pixel-interleaved inside it
LAYOUTS = {"bhw": "b (row h) (col w) -> row col (b h w)",
           "b":   "b (row h) (col w) -> row col b (h w)",
           "hwb": "b (row h) (col w) -> row col (h w b)"}

# rumi recipes: (recipe, tile_size, layout). The first two are the notebook's.
DEFAULT_RECIPES = [("planar>zigzag>pfor", 1024, "bhw"), ("med>zigzag>entropy", 1024, "bhw"),
                   ("planar>zigzag>zstd", 1024, "bhw"), ("planar>zigzag>pfor", 512, "bhw"),
                   ("planar>zigzag>pfor", 1024, "b")]

SECTIONS = ["env", "full", "config", "rumi", "window"]


# ----------------------------------------------------------------------------
# helpers

def log(*a):
    print(*a, flush=True)


def cpu_count():
    try:
        return len(os.sched_getaffinity(0))   # respects cgroup/taskset/Slurm
    except AttributeError:
        return os.cpu_count() or 1


def parse_threads(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        out.append(cpu_count() if tok in ("all", "all_cpus") else int(tok))
    return sorted(set(out))


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:12]


def to_bhw(a, nbands):
    """Normalise (h, w, b) or (h, w) outputs to (b, h, w)."""
    a = np.asarray(a)
    if a.ndim == 2:
        return a[None]
    if a.shape[-1] == nbands and a.shape[0] != nbands:
        return np.moveaxis(a, -1, 0)
    return a


def timeit(fn, reps, setup=None):
    """Median wall time of fn() over reps; returns (secs, last output of rep 0)."""
    times, first = [], None
    for i in range(reps):
        ctx = setup() if setup else None
        t0 = time.perf_counter()
        out = fn(ctx) if setup else fn()
        times.append(time.perf_counter() - t0)
        if i == 0:
            first = out
        del out
    return st.median(times), first


class Results:
    def __init__(self, raw_mb):
        self.rows = []
        self.raw_mb = raw_mb

    def add(self, section, engine, variant, case, threads, secs, ok, mb=None,
            size_mb=None, note=""):
        mb = self.raw_mb if mb is None else mb
        row = dict(section=section, engine=engine, variant=variant, case=case,
                   threads=threads, ms=None if secs is None else round(secs * 1000, 1),
                   mb_s=None if not secs else round(mb / secs),
                   size_mb=None if size_mb is None else round(size_mb, 1),
                   pixels_ok=ok, note=note)
        self.rows.append(row)
        ms = "-" if secs is None else f"{secs * 1000:9.1f}"
        rate = "-" if not secs else f"{mb / secs:7.0f}"
        flag = "" if ok in (True, None) else "  <-- PIXELS DIFFER"
        log(f"  {engine:15s} {variant:24s} {case:26s} {threads:3d} {ms:>9s} ms "
            f"{rate:>7s} MB/s{flag}{('  ' + note) if note else ''}")

    def skip(self, section, engine, variant, case, threads, why):
        self.rows.append(dict(section=section, engine=engine, variant=variant, case=case,
                              threads=threads, ms=None, mb_s=None, size_mb=None,
                              pixels_ok=None, note="SKIP: " + why))
        log(f"  {engine:15s} {variant:24s} {case:26s} {threads:3d}  skipped: {why}")

    def write_csv(self, path):
        if not self.rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)


# ----------------------------------------------------------------------------
# GDAL side

def gdal_setup(show_warnings=False):
    from osgeo import gdal
    gdal.UseExceptions()
    if not show_warnings:
        # e.g. GDAL 3.8 multithreaded reads of uncompressed pixel-interleaved
        # tiles emit one benign libtiff "allocChoppedUpStripArrays" per tile
        # warnings raised on GDAL worker threads bypass a pushed handler, so
        # route the default handler's output to the null device instead
        gdal.SetErrorHandler("CPLQuietErrorHandler")
        gdal.SetConfigOption("CPL_LOG", os.devnull)
    return gdal


def gdal_codecs(gdal):
    """Return (set of GTiff COMPRESS values, libdeflate?) from the driver metadata."""
    drv = gdal.GetDriverByName("GTiff")
    xml = drv.GetMetadataItem("DMD_CREATIONOPTIONLIST") or ""
    codecs = set()
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    libdeflate = False
    for opt in root.iter("Option"):
        if opt.get("name") == "COMPRESS":
            codecs = {v.text for v in opt.iter("Value")}
        if opt.get("name") == "ZLEVEL" and "12" in (opt.get("description") or ""):
            libdeflate = True
    return codecs, libdeflate


def gdal_open(gdal, path, driver, threads, open_opt=True):
    oo = [f"NUM_THREADS={threads}"] if open_opt else []
    return gdal.OpenEx(path, gdal.OF_RASTER | gdal.OF_READONLY,
                       allowed_drivers=[driver], open_options=oo)


def gdal_full_read(gdal, path, driver, threads, reps, cfg=None, reuse=False,
                   prealloc=False, interleave="band", open_opt=True):
    """Full-resolution read of all bands. Reopens the dataset every rep unless
    reuse=True (the notebook reuses one dataset, which lets GDAL_CACHEMAX serve
    repeats from the block cache when it is big enough)."""
    cfg = dict(cfg or {})
    cfg.setdefault("GDAL_NUM_THREADS", str(threads))
    cachemax = cfg.pop("GDAL_CACHEMAX", None)
    old_cache = gdal.GetCacheMax()
    if cachemax is not None:
        # process-wide setting: newer GDAL refuses it in thread-local config_options
        gdal.SetCacheMax(int(float(cachemax) * 1024 * 1024))
    try:
        with gdal.config_options(cfg):
            return _gdal_full_read(gdal, path, driver, threads, reps, reuse, prealloc,
                                   interleave, open_opt)
    finally:
        gdal.SetCacheMax(old_cache)


def _gdal_full_read(gdal, path, driver, threads, reps, reuse, prealloc, interleave,
                    open_opt):
    ds0 = gdal_open(gdal, path, driver, threads, open_opt)
    shape = (ds0.RasterCount, ds0.RasterYSize, ds0.RasterXSize)
    buf = None
    if prealloc:
        buf = np.empty(shape if interleave == "band" else
                       (shape[1], shape[2], shape[0]), np.uint8)

    def read(ds):
        if buf is not None:
            return ds.ReadAsArray(buf_obj=buf, interleave=interleave)
        return ds.ReadAsArray(interleave=interleave)

    if reuse:
        secs, out = timeit(lambda: read(ds0), reps)
    else:
        ds0 = None
        secs, out = timeit(read, reps,
                           setup=lambda: gdal_open(gdal, path, driver, threads, open_opt))
    return secs, to_bhw(out, shape[0])


def gdal_windows(gdal, path, driver, threads, reps, windows, band=None, cfg=None):
    cfg = dict(cfg or {})
    cfg.setdefault("GDAL_NUM_THREADS", str(threads))
    with gdal.config_options(cfg):
        def run(ds):
            outs = []
            for (r, c, h, w) in windows:
                if band is None:
                    outs.append(ds.ReadAsArray(c, r, w, h))
                else:
                    outs.append(ds.GetRasterBand(band + 1).ReadAsArray(c, r, w, h)[None])
            return outs
        return timeit(run, reps, setup=lambda: gdal_open(gdal, path, driver, threads))


# ----------------------------------------------------------------------------
# tifffile side

def tifffile_full_read(tif_bytes, threads, reps, nbands):
    import tifffile

    def run():
        return tifffile.imread(io.BytesIO(tif_bytes), key=0, maxworkers=threads)
    secs, out = timeit(run, reps)
    return secs, to_bhw(out, nbands)


# ----------------------------------------------------------------------------
# rumi side (encoding in-process, decoding in subprocess workers)

def rumi_encode(arr, recipe, tile, path, layout="bhw"):
    import geozl
    import rumi
    frames = rumi.frames(arr, LAYOUTS[layout], tile_size=tile)
    graphs = {}
    t0 = time.perf_counter()
    for f in frames:
        g = graphs.get(f.data.shape)
        if g is None:
            g = graphs[f.data.shape] = geozl.graph(f.data, recipe)
        f.compressed = geozl.compress(f.data, graph=g)
    enc = time.perf_counter() - t0
    rumi.write(path, frames)
    return enc


def rumi_worker(spec_path):
    """Runs in a fresh process: one (threads, verify) configuration."""
    spec = json.load(open(spec_path))
    import rumi
    rumi.set_num_threads(spec["threads"])
    rumi.set_checksum_verification(spec["verify"])
    got = rumi.get_num_threads(), rumi.get_checksum_verification()
    out = []
    for job in spec["jobs"]:
        path, reps, kind = job["path"], job["reps"], job["kind"]
        header = rumi.info(source=path).header
        src = open(path, "rb").read() if job["source"] == "bytes" else path
        if kind == "full":
            fn = lambda: rumi.read(src, header)
        elif kind == "windows":
            wins = [tuple(w) for w in job["windows"]]
            bands = job.get("bands")
            fn = lambda: [rumi.read(src, header, window=w, bands=bands) for w in wins]
        elif kind == "band":
            fn = lambda: rumi.read(src, header, bands=job["bands"])
        secs, res = timeit(fn, reps)
        if isinstance(res, list):
            dig = digest(np.concatenate([np.asarray(r).ravel() for r in res]))
        else:
            dig = digest(to_bhw(res, spec["nbands"]))
        out.append(dict(job, secs=secs, digest=dig))
    json.dump(dict(threads=got[0], verify=got[1], results=out), sys.stdout)


def rumi_run(jobs, threads, verify, nbands, workdir):
    spec = dict(threads=threads, verify=verify, nbands=nbands, jobs=jobs)
    sp = os.path.join(workdir, f"spec_{threads}_{int(verify)}.json")
    json.dump(spec, open(sp, "w"))
    env = dict(os.environ)
    env.pop("RUMI_NUM_THREADS", None)
    env.pop("RUMI_VERIFY", None)
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--_rumi-worker", sp],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout)


# ----------------------------------------------------------------------------
# main

def fetch(src, workdir):
    if os.path.exists(src):
        return src
    local = os.path.join(workdir, os.path.basename(src.split("?")[0]))
    if not os.path.exists(local):
        log(f"downloading {src}")
        with urllib.request.urlopen(src) as r, open(local + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(local + ".part", local)
    return local


def env_report(gdal, codecs, libdeflate):
    import geozl
    import rumi
    import tifffile
    log("== environment")
    log(f"  python      {platform.python_version()}  {platform.machine()}  "
        f"cpus(affinity)={cpu_count()} os.cpu_count={os.cpu_count()}")
    log(f"  numpy       {np.__version__}")
    log(f"  GDAL        {gdal.__version__}  (osgeo)  libdeflate={libdeflate}")
    log(f"  GTiff codecs {','.join(sorted(codecs))}")
    log(f"  LIBERTIFF   {'yes' if gdal.GetDriverByName('LIBERTIFF') else 'no (GDAL >= 3.11)'}")
    log(f"  rumi        {rumi.__version__}   geozl {geozl.__version__}   "
        f"simd {geozl.simd_info()}")
    try:
        import imagecodecs
        ic = imagecodecs.__version__
    except ImportError:
        ic = "not installed (tifffile falls back to zlib; zstd/lerc/jxl/webp unreadable)"
    log(f"  tifffile    {tifffile.__version__}   imagecodecs {ic}")
    for k in ("GDAL_CACHEMAX", "GDAL_NUM_THREADS", "RUMI_NUM_THREADS", "RUMI_VERIFY",
              "OMP_NUM_THREADS"):
        if os.environ.get(k):
            log(f"  env {k}={os.environ[k]}")


def run_children(a, src, workdir, threads_list, sections):
    """One process per thread count, so GDAL's never-shrinking global thread
    pool (and rumi's pinned pool) reflect only that count."""
    tmax = max(threads_list)
    rows = []
    for t in threads_list:
        sec = sorted(sections - {"env"} - ({"config"} if t != tmax else set()))
        if not sec:
            continue
        log(f"\n######## {t} thread(s): {','.join(sec)}")
        child_csv = os.path.join(workdir, f"child_t{t}.csv")
        cmd = [sys.executable, os.path.abspath(__file__), "--_child",
               "--src", src, "--workdir", workdir, "--threads", str(t),
               "--reps", str(a.reps), "--variants", a.variants,
               "--sections", ",".join(sec), "--nwin", str(a.nwin),
               "--winsize", str(a.winsize), "--cachemax", a.cachemax, "--csv", child_csv]
        if a.recipes:
            cmd += ["--recipes", a.recipes]
        if a.gdal_warnings:
            cmd += ["--gdal-warnings"]
        for kv in a.config:
            cmd += ["--config", kv]
        env = dict(os.environ)
        env.pop("GDAL_NUM_THREADS", None)
        if subprocess.run(cmd, env=env).returncode != 0:
            log(f"  child for {t} thread(s) failed")
            continue
        with open(child_csv, newline="") as f:
            rows += list(csv.DictReader(f))
    out = a.csv if os.path.isabs(a.csv) else os.path.join(workdir, a.csv)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    bad = [r for r in rows if r["pixels_ok"] == "False"]
    log(f"\nwrote {out}  ({len(rows)} rows, {len(bad)} pixel mismatches)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_URL, help="GeoTIFF path or URL")
    ap.add_argument("--workdir", default="rumi_bench_work")
    ap.add_argument("--threads", default="1,all", help="e.g. 1,2,4,all")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                    help="comma list, or 'all'. Known: orig," + ",".join(VARIANTS))
    ap.add_argument("--recipes", default=None,
                    help="comma list of recipe@tile[@layout], layout one of bhw (default), "
                         "b, hwb; e.g. 'planar>zigzag>pfor@1024@b,med>zigzag>entropy@512'")
    ap.add_argument("--sections", default=",".join(SECTIONS),
                    help="comma list of " + ",".join(SECTIONS))
    ap.add_argument("--nwin", type=int, default=16, help="windows per window test")
    ap.add_argument("--winsize", type=int, default=1024)
    ap.add_argument("--config", action="append", default=[],
                    help="extra GDAL config KEY=VAL, benchmarked as its own case "
                         "(repeatable)")
    ap.add_argument("--cachemax", default="16,64,512,4096",
                    help="GDAL_CACHEMAX values (MB) for the config sweep")
    ap.add_argument("--csv", default="rumi_bench_results.csv")
    ap.add_argument("--gdal-warnings", action="store_true",
                    help="show GDAL/libtiff warnings (quiet by default)")
    ap.add_argument("--profile", action="store_true",
                    help="run geozl.profile on one tile to list candidate recipes, then exit")
    ap.add_argument("--_rumi-worker", dest="rumi_worker", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--_child", dest="child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.rumi_worker:
        rumi_worker(a.rumi_worker)
        return

    sections = set(a.sections.split(","))
    threads_list = parse_threads(a.threads)
    tmax = max(threads_list)
    os.makedirs(a.workdir, exist_ok=True)
    workdir = os.path.abspath(a.workdir)

    gdal = gdal_setup(a.gdal_warnings)
    codecs, libdeflate = gdal_codecs(gdal)
    have_libertiff = gdal.GetDriverByName("LIBERTIFF") is not None
    drivers = ["GTiff"] + (["LIBERTIFF"] if have_libertiff else [])
    if "env" in sections and not a.child:
        env_report(gdal, codecs, libdeflate)

    # ---- source
    src = fetch(a.src, workdir)
    # GDAL's global thread pool grows to the largest count ever requested and
    # never shrinks, so a child process only ever asks for its own count
    with gdal.config_options({"GDAL_NUM_THREADS": str(tmax) if a.child else "ALL_CPUS"}):
        ds = gdal.Open(src)
        arr = ds.ReadAsArray()
        blk = ds.GetRasterBand(1).GetBlockSize()
        md = ds.GetMetadata("IMAGE_STRUCTURE")
        if not a.child:
            log(f"\n== source {src}\n  {arr.shape} {arr.dtype} block {blk} {md} "
                f"overviews {ds.GetRasterBand(1).GetOverviewCount()}")
        ds = None
    nb = arr.shape[0]
    ref = digest(arr)
    raw_mb = arr.nbytes / 1e6
    R = Results(raw_mb)
    if not a.child:
        log(f"  uncompressed {raw_mb:.1f} MB  reference digest {ref}")

    if a.profile:
        import geozl
        t = arr[:, :1024, :1024]
        for prior in ("planar", "med"):
            try:
                log(geozl.profile(t, prior=prior, reps=3))
            except Exception as e:  # noqa: BLE001
                log(f"  prior {prior}: {e}")
        return

    # ---- GTiff variants (no overviews, same pixels): the parent writes them to
    # disk, each child loads them into /vsimem
    import tifffile
    with tifffile.TiffFile(src) as tf:
        orig_full_mb = sum(tf.pages[0].databytecounts) / 1e6
    vdir = os.path.join(workdir, "variants")
    os.makedirs(vdir, exist_ok=True)
    wanted = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    wanted = [w for w in wanted if w != "orig"]
    if not a.child:
        for name in wanted:
            out = os.path.join(vdir, name + ".tif")
            if os.path.exists(out):
                os.remove(out)
            if name not in VARIANTS:
                log(f"  unknown variant {name}")
                continue
            co = list(VARIANTS[name])
            codec = co[0].split("=")[1]
            if codec not in codecs:
                log(f"  variant {name}: GTiff here has no {codec}, skipped")
                continue
            bs = name.rsplit("-", 1)[1]
            co += ["TILED=YES", f"BLOCKXSIZE={bs}", f"BLOCKYSIZE={bs}",
                   "NUM_THREADS=ALL_CPUS"]
            try:
                gdal.Translate(out, src, format="GTiff", creationOptions=co)
            except RuntimeError as e:
                log(f"  variant {name}: {e}")
    tif_bytes = {"orig": open(src, "rb").read()}
    vpath = {"orig": "/vsimem/orig.tif"}
    vsize = {"orig": orig_full_mb}
    for name in wanted:
        f = os.path.join(vdir, name + ".tif")
        if os.path.exists(f):
            tif_bytes[name] = open(f, "rb").read()
            vpath[name] = f"/vsimem/{name}.tif"
            vsize[name] = len(tif_bytes[name]) / 1e6
    for name, p in vpath.items():
        gdal.FileFromMemBuffer(p, tif_bytes[name])
    if not a.child:
        log("\n== sizes (full resolution only)")
        for name, mb in vsize.items():
            log(f"  {name:22s} {mb:7.1f} MB  "
                f"{100 * (mb - orig_full_mb) / orig_full_mb:+6.1f}% vs orig")

    # ---- rumi encodes
    recipes = DEFAULT_RECIPES
    if a.recipes:
        recipes = []
        for r in a.recipes.split(","):
            parts = r.split("@")
            recipes.append((parts[0], int(parts[1]) if len(parts) > 1 else 1024,
                            parts[2] if len(parts) > 2 else "bhw"))
    rumi_files = {}
    if sections & {"rumi", "window", "full"}:
        for recipe, tile, layout in recipes:
            key = f"{recipe}@{tile}" + ("" if layout == "bhw" else f"@{layout}")
            path = os.path.join(workdir, key.replace(">", "_").replace("@", "_t") + ".rumi")
            if a.child:
                if os.path.exists(path):
                    rumi_files[key] = path
                    vsize["rumi " + key] = os.path.getsize(path) / 1e6
                continue
            if os.path.exists(path):
                os.remove(path)
            try:
                enc = rumi_encode(arr, recipe, tile, path, layout)
            except Exception as e:  # noqa: BLE001
                log(f"  rumi {key}: encode failed: {e}")
                continue
            rumi_files[key] = path
            mb = os.path.getsize(path) / 1e6
            vsize["rumi " + key] = mb
            log(f"  rumi {key:28s} {mb:7.1f} MB  {100 * (mb - orig_full_mb) / orig_full_mb:+6.1f}%"
                f" vs orig  (encode {enc:.1f}s)")

    if not a.child:
        run_children(a, src, workdir, threads_list, sections)
        return

    def ok(d):
        return d == ref

    # ---- section: full reads, every engine x variant x threads
    if "full" in sections:
        log("\n== full-resolution read, all bands (dataset reopened each rep)")
        for name, p in vpath.items():
            for t in threads_list:
                for drv in drivers:
                    try:
                        secs, out = gdal_full_read(gdal, p, drv, t, a.reps)
                        R.add("full", "GDAL " + drv, name, "default", t, secs,
                              ok(digest(out)), size_mb=vsize[name])
                    except Exception as e:  # noqa: BLE001
                        R.skip("full", "GDAL " + drv, name, "default", t, str(e)[:80])
                try:
                    secs, out = tifffile_full_read(tif_bytes[name], t, a.reps, nb)
                    R.add("full", "tifffile", name, "default", t, secs, ok(digest(out)),
                          size_mb=vsize[name])
                except Exception as e:  # noqa: BLE001
                    R.skip("full", "tifffile", name, "default", t, str(e)[:80])

    # ---- section: GDAL config sweeps on orig (and uncompressed variants)
    if "config" in sections:
        log(f"\n== GDAL config sweeps (orig, {tmax} threads unless noted)")
        p = vpath["orig"]
        cases = []
        # thread mechanism: open option only / config only / both
        cases += [("threads: open-opt only", dict(cfg={"GDAL_NUM_THREADS": "1"}), tmax),
                  ("threads: config only", dict(cfg={"GDAL_NUM_THREADS": str(tmax)},
                                                open_opt=False), tmax),
                  ("threads: ALL_CPUS cfg", dict(cfg={"GDAL_NUM_THREADS": "ALL_CPUS"},
                                                 open_opt=False), tmax)]
        for cm in a.cachemax.split(","):
            cases.append((f"GDAL_CACHEMAX={cm}", dict(cfg={"GDAL_CACHEMAX": cm}), tmax))
        # the cache trap: reusing one dataset with a cache bigger than the image
        big = str(int(raw_mb * 2) + 64)
        cases += [(f"reuse ds, CACHEMAX={big}", dict(cfg={"GDAL_CACHEMAX": big}, reuse=True),
                   tmax),
                  ("reuse ds, CACHEMAX=64", dict(cfg={"GDAL_CACHEMAX": "64"}, reuse=True), tmax),
                  ("prealloc buffer", dict(prealloc=True), tmax),
                  ("pixel-interleaved out", dict(interleave="pixel"), tmax),
                  ("pixel out + prealloc", dict(interleave="pixel", prealloc=True), tmax)]
        for kv in a.config:
            k, v = kv.split("=", 1)
            cases.append((f"{k}={v}"[:26], dict(cfg={k: v}), tmax))
        for drv in drivers:
            for label, kw, t in cases:
                try:
                    secs, out = gdal_full_read(gdal, p, drv, t, a.reps, **kw)
                    R.add("config", "GDAL " + drv, "orig", label, t, secs, ok(digest(out)))
                except Exception as e:  # noqa: BLE001
                    R.skip("config", "GDAL " + drv, "orig", label, t, str(e)[:80])
        # disk vs /vsimem (disk file is page-cache warm after the first rep)
        for drv in drivers:
            secs, out = gdal_full_read(gdal, src, drv, tmax, a.reps)
            R.add("config", "GDAL " + drv, "orig", "local file (page cache)", tmax, secs,
                  ok(digest(out)))
        # uncompressed paths
        for name in ("none-band-1024", "none-pixel-1024"):
            if name not in vpath:
                continue
            for label, cfg in (("default", {}),
                               ("GTIFF_DIRECT_IO=YES", {"GTIFF_DIRECT_IO": "YES"}),
                               ("GTIFF_VIRTUAL_MEM_IO=YES", {"GTIFF_VIRTUAL_MEM_IO": "YES"})):
                try:
                    secs, out = gdal_full_read(gdal, vpath[name], "GTiff", tmax, a.reps, cfg=cfg)
                    R.add("config", "GDAL GTiff", name, label, tmax, secs, ok(digest(out)))
                except Exception as e:  # noqa: BLE001
                    R.skip("config", "GDAL GTiff", name, label, tmax, str(e)[:80])

    # ---- windows (shared by GDAL and rumi)
    H, W = arr.shape[1:]
    ws = a.winsize
    rng = np.random.default_rng(42)

    def pick(offset):
        rows = np.arange(offset, H - ws + 1, ws)
        cols = np.arange(offset, W - ws + 1, ws)
        cells = [(int(r), int(c)) for r in rows for c in cols]
        idx = rng.choice(len(cells), size=min(a.nwin, len(cells)), replace=False)
        return [(cells[i][0], cells[i][1], ws, ws) for i in sorted(idx)]

    win_sets = {"aligned": pick(0), "offset-half": pick(ws // 2)}
    win_ref = {k: digest(np.concatenate([arr[:, r:r + h, c:c + w].ravel()
                                         for r, c, h, w in v])) for k, v in win_sets.items()}
    win_ref_b0 = {k: digest(np.concatenate([arr[0:1, r:r + h, c:c + w].ravel()
                                            for r, c, h, w in v])) for k, v in win_sets.items()}
    win_mb = {k: len(v) * nb * ws * ws / 1e6 for k, v in win_sets.items()}

    # ---- section: rumi (subprocess per threads x verify)
    if sections & {"full", "rumi", "window"}:
        log("\n== rumi (fresh process per thread count / verify setting)")
        for t in threads_list:
            for verify in ((False, True) if "rumi" in sections else (False,)):
                jobs = []
                for key, path in rumi_files.items():
                    if sections & {"full", "rumi"}:
                        jobs.append(dict(key=key, path=path, reps=a.reps, kind="full",
                                         source="bytes"))
                    if "rumi" in sections:
                        if not verify:
                            jobs.append(dict(key=key, path=path, reps=a.reps, kind="full",
                                             source="path"))
                            jobs.append(dict(key=key, path=path, reps=a.reps, kind="band",
                                             bands=[0], source="bytes"))
                    if "window" in sections and not verify:
                        for wk, wv in win_sets.items():
                            jobs.append(dict(key=key, path=path, reps=a.reps, kind="windows",
                                             windows=wv, source="bytes", wset=wk))
                            jobs.append(dict(key=key, path=path, reps=a.reps, kind="windows",
                                             windows=wv, source="bytes", wset=wk, bands=[0]))
                if not jobs:
                    continue
                try:
                    res = rumi_run(jobs, t, verify, nb, workdir)
                except RuntimeError as e:
                    R.skip("rumi", "rumi", "-", f"verify={verify}", t, str(e)[-80:])
                    continue
                if res["threads"] != t or res["verify"] != verify:
                    log(f"  WARNING rumi reports threads={res['threads']} "
                        f"verify={res['verify']}")
                for j in res["results"]:
                    eng = "rumi"
                    var = j["key"].replace(">zigzag>", ">")
                    if j["kind"] == "full":
                        case = f"{j['source']}, verify={verify}"
                        R.add("rumi" if verify or j["source"] == "path" else "full",
                              eng, var, case, t, j["secs"], ok(j["digest"]),
                              size_mb=vsize["rumi " + j["key"]])
                    elif j["kind"] == "band":
                        R.add("window", eng, var, "band 1 only, full extent", t, j["secs"],
                              j["digest"] == digest(arr[0:1]), mb=raw_mb / nb)
                    else:
                        b = j.get("bands")
                        refd = (win_ref_b0 if b else win_ref)[j["wset"]]
                        mb = win_mb[j["wset"]] / (nb if b else 1)
                        R.add("window", eng, var,
                              f"{len(win_sets[j['wset']])}x{ws} {j['wset']}{' b1' if b else ''}",
                              t, j["secs"], j["digest"] == refd, mb=mb)

    # ---- section: GDAL windows / single band
    if "window" in sections:
        log(f"\n== windows ({a.nwin} x {ws}px, aligned to tile grid and offset by half)")
        for name in [v for v in ("orig", "deflate-band-1024", "zstd-band-1024") if v in vpath]:
            for t in threads_list:
                for drv in drivers:
                    for wk, wv in win_sets.items():
                        for band in (None, 0):
                            try:
                                secs, outs = gdal_windows(gdal, vpath[name], drv, t, a.reps,
                                                          wv, band=band)
                                d = digest(np.concatenate([np.asarray(o).ravel()
                                                           for o in outs]))
                                refd = (win_ref if band is None else win_ref_b0)[wk]
                                mb = win_mb[wk] / (1 if band is None else nb)
                                R.add("window", "GDAL " + drv, name,
                                      f"{len(wv)}x{ws} {wk}{' b1' if band is not None else ''}",
                                      t, secs, d == refd, mb=mb)
                            except Exception as e:  # noqa: BLE001
                                R.skip("window", "GDAL " + drv, name, wk, t, str(e)[:80])
                    # single band, full extent
                    try:
                        with gdal.config_options({"GDAL_NUM_THREADS": str(t)}):
                            secs, out = timeit(
                                lambda ds: ds.GetRasterBand(1).ReadAsArray(), a.reps,
                                setup=lambda: gdal_open(gdal, vpath[name], drv, t))
                        R.add("window", "GDAL " + drv, name, "band 1 only, full extent", t,
                              secs, digest(out) == digest(arr[0]), mb=raw_mb / nb)
                    except Exception as e:  # noqa: BLE001
                        R.skip("window", "GDAL " + drv, name, "band 1", t, str(e)[:80])

    R.write_csv(os.path.join(workdir, a.csv) if not os.path.isabs(a.csv) else a.csv)


if __name__ == "__main__":
    main()
