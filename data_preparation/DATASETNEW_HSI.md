# DatasetNew HSI crown-instance export

Run from the TreeMonitor repository root:

```powershell
& .\.venv\Scripts\python.exe code\BalSAM\data_preparation\build_datasetnew_coco.py
```

The script reads the two `raw/orthophoto/*_hsi.tif` images, corresponding
`raw/dsm/*_dsm.tif` DSMs, `raw/labels/*_crowns_new.gpkg` crown polygons, and
the saved split GeoPackages under `Data/DatasetNew/splits/`. It does not alter
those source files. Output goes to
`Data/DatasetNew/processed/balsam_hsi_512_stride256/`.

- `train` (source AOI `train_gold`): 1024-by-1024 source-pixel crops resampled to 512-by-512 outputs at source stride 256, entirely inside
  the saved gold AOI. Its polygons are the supervised ground truth.
- `val` and `test`: the exact fixed, non-overlapping 1024-by-1024 source-grid tiles
  in the saved spatial split. Their existing crown polygons are exported, but
  are provisional because they are not exhaustively audited.
  Complete/audit these areas before using AP or similar metrics.

Each split (`train`, `val`, `test`) has `hsi/*.tif` (164-band uint16, half the original linear resolution), aligned
`dsm/*.tif` (single-band float32), and for annotated splits `coco/*.json`
(one COCO file per tile). Combined split annotations are in
`annotations/{train,val,test}.json`. All instances have
one category, `tree`; no species or family class is inferred. `manifest.json`
records source site and pixel offsets, including any skipped low-validity
windows. `summary.json` gives counts and annotation warnings.

The HSI is average-resampled from each 1024-by-1024 source window to a
512-by-512 grid. DSM is bilinearly reprojected to that same output grid, and
COCO polygons are transformed to the corresponding 512-pixel coordinates. All
outputs preserve the same geographic footprint as the source window. DSM values
represent surface elevation, not canopy height. A custom hyperspectral
loader/adapter is required for BalSAM, whose stock image loader expects RGB data.

Use `--manifest-only --output <different-path>` for a dry run, or
`--limit 1 --output <different-path>` for a single-tile export check. An
interrupted export may be restarted with `--resume-partial` at the same output
path; existing generated tiles there are overwritten.
