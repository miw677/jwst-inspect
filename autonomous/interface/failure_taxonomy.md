# Failure-mode taxonomy (template)

Catalog the ways inspection policies fail, especially failures that appear only
under path-traced evaluation (the R2P story). Each entry: trigger, observable,
metric impact, example episode id. Numbers/examples come from eval artifacts.

## Perception-driven
- Specular glare saturation on the gold mirror -> lost feature track -> standoff drift.
- Sunshield specular highlight mistaken for structure -> false anomaly.
- Depth noise at range -> premature/late standoff correction.

## Control-driven
- Keep-out near-miss under latency -> abort or violation.
- Relative-velocity overshoot during approach -> corridor exit.
- Thruster saturation -> incomplete survey coverage.

## R2P-specific (rasterized-trained, path-traced-evaluated)
- Policy overfit to rasterized lighting -> degraded coverage under path tracing.
- Reflection/secondary-bounce cues absent in raster -> misjudged geometry in PT.

## Evaluation/process
- Seed leakage between train and eval splits.
- Metric instability across seeds (report confidence intervals).
