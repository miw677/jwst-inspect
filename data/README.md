# The shared data

This folder is the project's dataset mirror: the same public NASA/STScI data
that lives at `/data/shared/raw` on the workstation, tracked here (via Git LFS)
so every team member can see exactly what the project works with, where each
file came from, and how to verify it - even without a workstation session.

On the workstation, always read from `/data/shared/raw` (canonical, already on
fast local disk, read-only). Clones on the box skip the LFS download, so the
files here appear as small pointer files; that is intentional. Off the box,
`git lfs pull -I data/<path>` materializes any file, and `fetch_missing.sh`
downloads the datasets that are too large to track at all.

## What is here

Sizes are the on-disk footprint measured at import (2026-07-04); per-file
bytes, sha256, source URL, and retrieved-at live in `manifest.csv`.

| Folder | What it is | Files | Size | License |
|---|---|---|---|---|
| `jwst_geometry/` | NASA JWST CAD: detailed model A (Maya source + real gold/sunshield texture maps + `nasa-jwst-model-a.usdc`), model B GLB seed, 3D-print STL/USDZ kit, curated visual-reference CSV | 36 | 0.42 GiB | NASA media usage / US Gov public domain |
| `jwst_ephemeris/` | Real JPL Horizons JWST state vectors, sun- and earth-centered, 2022-2026 at 6 h steps (7,301 states per center) | 2 | 4 MiB | JPL Horizons public |
| `env_lighting/` | NASA SVS Deep Star Maps 2020 (8k/16k EXR), TSIS-1 solar reference spectrum, measured Au/Al/Si optical constants (refractiveindex.info) | 9 | 0.80 GiB | NASA SVS public / LASP LISIRD public / refractiveindex.info CC0 |
| `jwst_imagery_fits/` | 137 calibrated JWST images (MIRI + NIRCam i2d: NGC 3132, WR 140, SMACS J0723, Stephan's Quintet) + the official ERO download script | 138 | 5.91 GiB | STScI public |
| `inspector_refs/` | JWST commissioning science-performance report (arXiv 2207.05632, incl. the real C3 micrometeoroid strike) + curated inspector-heritage reference CSV | 2 | 13 MiB | arXiv open access / NASA NTRS public |
| `manifest.csv` | Provenance for every downloaded file: dataset, relpath, bytes, sha256, source URL, retrieved-at UTC | 1 | 39 KB | project artifact |

## What is not here, and why

The GitLab free tier hard-caps a project at 10 GiB of repository plus LFS;
past it the project turns read-only. The full raw tree is above that, so the
two largest datasets stay workstation-only and are fully described by
`manifest.csv`:

| Dataset | Size | On the workstation | Anywhere else |
|---|---|---|---|
| `speed_plus/` | 15.8 GiB zip, 31.9 GiB extracted (69,499 files) - Stanford SLAB SPEED+ v2 spacecraft pose benchmark, CC BY 4.0, Zenodo record 5588480 | `/data/shared/raw/speed_plus` | `./fetch_missing.sh` |
| `ifu_cubes/` | 3.04 GiB - 23 JWST MIRI MRS IFU spectral cubes (WR 140, Stephan's Quintet, Cas A) for volumetric backgrounds | `/data/shared/raw/ifu_cubes` | `./fetch_missing.sh` |

```bash
# Recreate an excluded dataset anywhere (URL + sha256 come from manifest.csv):
FETCH_DATASETS=speed_plus FETCH_DEST_ROOT=/some/path ./fetch_missing.sh
```

The synthetic benchmark dataset (Group 2's Replicator output, hundreds of GB)
is generated, not downloaded; it lands in `/data/shared/datasets` on the
workstation with its own manifest and never enters git.

## Verifying a file

Every row of `manifest.csv` is checkable:

```bash
# pick a row, then:
sha256sum /data/shared/raw/<relpath>     # must equal the manifest sha256
```

## Loading examples

```python
# FITS image (conda env: jwst-astro)
from astropy.io import fits
hdul = fits.open("/data/shared/raw/jwst_imagery_fits/ngc3132/"
                 "hlsp_jwst-ero_jwst_miri_ngc3132_f770w_v1_i2d.fits")
sci = hdul["SCI"].data                      # 2D calibrated image, MJy/sr

# Ephemeris (env: jwst-base or jwst-astro)
import pandas as pd
eph = pd.read_csv("/data/shared/raw/jwst_ephemeris/"
                  "jwst_horizons_vectors_sun_2022-01-01_2026-12-31_6h.csv")

# USD stage (env: jwst-usd, or inside the Isaac Sim container)
from pxr import Usd
stage = Usd.Stage.Open("/data/shared/raw/jwst_geometry/model_a/nasa-jwst-model-a.usdc")
```

## Changing the data

`/data/shared/raw` is read-only for everyone but the admin, and this mirror
tracks it exactly. Need a new dataset, more files from an existing source, or
a correction? Ask the admin: the download pipeline re-runs with new
parameters, `manifest.csv` is regenerated, and the mirror updates through a
merge request on the main project so every fork sees the same change.
