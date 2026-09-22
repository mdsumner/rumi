# rumi vs GeoTIFF: decode benchmark

This benchmark extends the rumi notebook
[asterisk-labs/rumi examples/rumi-vs-geotiff.ipynb](https://github.com/asterisk-labs/rumi/blob/main/examples/rumi-vs-geotiff.ipynb) that was [posted on Pangeo](https://discourse.pangeo.io/t/decode-geotiff-to-gpu-memory/5214/16). 

It splits rumi's speed-up over a Sentinel-2 COG into its parts: the codec, the
band layout inside a chunk, the tile size, and the GDAL settings.

Scene: `S2A_37MBV_20241029_0_L2A/TCI.tif` (3 x 10980 x 10980 uint8, DEFLATE +
PREDICTOR=2, pixel-interleaved, 1024 tiles). The encoded bytes are held in RAM,
so only decoding is timed. Each figure is the median of 5 reads. Every output
is hashed against the source pixels, and all rows matched.

## What rumi is

A rumi file has a BigTIFF IFD, with `RUMI` in place of the `II+\0` magic bytes,
and a single full-resolution tile index (no overviews). Each tile is an
[OpenZL](https://github.com/facebook/openzl) frame, and a private tag records
what one chunk holds:

- all bands in one chunk, stored band after band (`bhw`, the notebook default)
- one chunk per band (`b`, like TIFF `PlanarConfiguration=2`)
- all bands in one chunk, pixel-interleaved (`hwb`)

Decoding runs on the CPU, one thread per tile. Output goes to
numpy/torch/jax/tf through DLPack. `read_many()` reads same-sized windows from
many files into one batch, so the format is aimed at feeding an ML data loader.

## Results (full read, all bands, ms; GDAL = LIBERTIFF)

| encoding | MB | 1 thr | 4 thr | 16 thr | 32 thr |
|---|---:|---:|---:|---:|---:|
| orig COG (deflate, pixel-interleaved) | 237 | 1199 | 326 | 93 | 73 |
| deflate, band-interleaved, 1024 tiles | 234 | 1159 | 302 | 93 | 51 |
| zstd, band-interleaved, 1024 tiles | 230 | 622 | 166 | 55 | 35 |
| uncompressed, band-interleaved (floor) | 381 | 85 | 38 | 27 | 20 |
| rumi pfor, 1024 tiles, bhw | 250 | 439 | 136 | 55 | 67 |
| rumi pfor, 1024 tiles, one chunk per band | 250 | 444 | 119 | 43 | 33 |
| rumi pfor, 512 tiles, bhw | 250 | 442 | 127 | 50 | 42 |
| rumi zstd, 1024 tiles, bhw | 224 | 766 | 221 | 93 | 101 |

Band 1 only, 16 x 1024px windows aligned to the tile grid, 1 thread:
rumi one-chunk-per-band **9 ms**, GDAL zstd band-interleaved 25 ms, rumi bhw
(the notebook default) 36 ms, orig COG 163 ms.

## Take-home

1. **The codec accounts for most of the gap, not the layout.** Rewriting the
   COG as band-interleaved DEFLATE changes almost nothing (1199 vs 1159 ms).
   Switching to band-interleaved ZSTD in a plain GTiff halves the time, and
   the file is smaller.
2. **pfor's real advantage is about 1.4x on one thread** over the best
   GeoTIFF any GDAL reader can already open. The file is 8% bigger. With 16
   or more threads, both approach the memory-bandwidth floor and the gap
   mostly closes. rumi's own zstd recipe is slower than GDAL's ZSTD.
3. **One chunk per band helps band subsets and windows.** It gives the same
   gain any band-interleaved GTiff gets, and rumi's default layout doesn't
   use it.
4. **Tile size matters for windows that aren't tile-aligned.** Windows offset
   by half a tile cost about 2.5x more with 1024 tiles than with 512.
   (GDAL was not timed on 512-tile windows in this run.)
5. **Most GDAL settings make little difference here.** Setting threads by
   open option or config option makes no difference, and neither does
   `GDAL_CACHEMAX`. A preallocated buffer plus pixel-interleaved output gains
   about 10-20%. `GTIFF_DIRECT_IO` and `GTIFF_VIRTUAL_MEM_IO` make in-memory
   reads 3-4x slower.

New codecs deserve as much effort on being readable from
existing software (a TIFF compression tag, a libtiff/GDAL codec, an
imagecodecs/numcodecs entry) as on the codec itself. rumi is already a TIFF
in all but its magic bytes. Its OpenZL codec inside a standard GTiff would
be readable by every TIFF reader.

## Reproduce

In the gdal-r-ci `extras` image (system GDAL through `osgeo.gdal`; numpy kept
at the version the bindings were built with):

```
python -c "import numpy; print('numpy==' + numpy.__version__)" > /tmp/np.txt
uv pip install -c /tmp/np.txt "rumi-eo[write]" tifffile imagecodecs

python rumi_bench.py --threads 1,4,16,32 \
  --variants orig,deflate-band-1024,zstd-band-1024,deflate-band-512,none-band-1024 \
  --recipes "planar>zigzag>pfor@1024,planar>zigzag>pfor@1024@b,planar>zigzag>pfor@512,planar>zigzag>zstd@1024"
```

`python rumi_bench.py --help` lists every variant, recipe, layout and section.
Results go to `rumi_bench_work/rumi_bench_results.csv`.

Method notes:

- **Separate processes per thread count.** GDAL's global thread pool never
  shrinks, and rumi fixes its thread count and checksum setting at the first
  read. So each thread count runs in its own process, and rumi also gets a
  fresh process per checksum setting. Results from before this fix
  overstated GDAL's scaling at 4 or more threads.
- **Dataset reopened every repeat,** so GDAL's block cache can't serve a repeat
  read.
- **tifffile is included for completeness.** It is much slower multithreaded.
- **Scene-dependent.** These are one scene's numbers. Another scene, dtype or
  machine will differ, so the environment block the script prints first
  should be kept with the results.


### Full run

```
docker run --rm -ti -v $(pwd):/rumi ghcr.io/hypertidy/gdal-r-python-extras:latest
#── ghcr.io/hypertidy/gdal-r-python-extras ──
#GDAL 3.13.3   PROJ 9.9.0   GEOS 3.15.0
#R    4.6.1   (kitchen sink loaded)
#Py   3.12.3

#Docs: https://github.com/hypertidy/gdal-r-ci#gdal-r-python-extras
cd /rumi
python -c "import numpy; print('numpy==' + numpy.__version__)" > /tmp/np.txt
uv pip install -c /tmp/np.txt "rumi-eo[write]" tifffile imagecodecs
#Using Python 3.12.3 environment at: /opt/gdal-py
#Resolved 8 packages in 247ms
#Prepared 3 packages in 164ms
#Installed 3 packages in 79ms
# + geozl==0.18.0
# + openzl==0.2.0
# + rumi-eo==0.24.0
python rumi_bench.py --threads 1,4,16,32 --variants orig,deflate-band-1024,zstd-band-1024,deflate-band-512,none-band-1024 --recipes "planar>zigzag>pfor@1024,planar>zigzag>pfor@1024@b,planar>zigzag>pfor@512,planar>zigzag>zstd@1024"
```

```
== environment
  python      3.12.3  x86_64  cpus(affinity)=32 os.cpu_count=32
  numpy       2.5.3
  GDAL        3.13.3  (osgeo)  libdeflate=True
  GTiff codecs CCITTFAX3,CCITTFAX4,CCITTRLE,DEFLATE,JPEG,LERC,LERC_DEFLATE,LERC_ZSTD,LZMA,LZW,NONE,PACKBITS,WEBP,ZSTD
  LIBERTIFF   yes
  rumi        0.24.0   geozl 0.18.0   simd {'built': ['scalar', 'sse2', 'avx2'], 'cpu': ['scalar', 'sse2', 'avx2'], 'active': 'avx2'}
  tifffile    2026.9.20   imagecodecs 2026.8.16
downloading https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/37/M/BV/2024/10/S2A_37MBV_20241029_0_L2A/TCI.tif

== source /rumi/rumi_bench_work/TCI.tif
  (3, 10980, 10980) uint8 block [1024, 1024] {'COMPRESSION': 'DEFLATE', 'INTERLEAVE': 'PIXEL', 'PREDICTOR': '2', 'LAYOUT': 'COG'} overviews 4
  uncompressed 361.7 MB  reference digest a0a0ec0342de

== sizes (full resolution only)
  orig                     236.7 MB    +0.0% vs orig
  deflate-band-1024        234.0 MB    -1.1% vs orig
  zstd-band-1024           230.4 MB    -2.6% vs orig
  deflate-band-512         232.8 MB    -1.7% vs orig
  none-band-1024           380.6 MB   +60.8% vs orig
  rumi planar>zigzag>pfor@1024        249.8 MB    +5.5% vs orig  (encode 2.3s)
  rumi planar>zigzag>pfor@1024@b      249.8 MB    +5.5% vs orig  (encode 2.2s)
  rumi planar>zigzag>pfor@512         249.7 MB    +5.5% vs orig  (encode 2.2s)
  rumi planar>zigzag>zstd@1024        224.2 MB    -5.3% vs orig  (encode 5.7s)
```
