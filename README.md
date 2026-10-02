# Forecast Calibration and Route-Choice Utility: Core Evidence

This small release contains only the frozen tables and day-level aggregates needed to check the paper's central empirical comparisons. It is a **result-level verification package**, not a full raw-data or model-training release.

## Main results

| Formal comparison | Joint coverage, C0 → C2 | Lateness, C2 − C0 | Mean travel time, C2 − C0 |
|---|---:|---:|---:|
| Beijing M1 | 83.19% → 92.26% | +0.1633 percentage points | +0.588 s |
| Chengdu M1 | 75.14% → 88.33% | +0.7848 percentage points | +3.082 s |
| Chengdu M2 | 74.01% → 90.64% | +0.9200 percentage points | +4.418 s |

In a separate within-Chengdu predictor comparison, M2 has 14.91% lower speed MAE than M0 and 1.4571 percentage points lower lateness under the same raw-bound C0 route interface. All route-decision results concern frozen offline proxy tasks on publicly processed speed data, not observed online navigation benefits.

## Files

- `data/dataset_scope.csv` — road and task scope, including repeated observation counts.
- `data/accuracy_decision.csv` — within-city speed error and C0 decision outcomes.
- `data/primary_comparisons.csv` — absolute coverage, upper-score expansion, paired decision effects, and intervals.
- `data/decision_interfaces.csv` — route interfaces, simple baselines, and post hoc oracle headroom. Beijing interface rows are post-result supplementary analyses.
- `data/switch_decomposition.csv` — formal versus supplemental route-switch accounting. The Beijing three-seed formal total of 576 net harmful switches is separate from the E168-CDF supplemental count of 155.
- `data/beijing_daily_paired.csv` and `data/beijing_bootstrap_indices.npz` — Beijing date-level paired effects and the frozen non-circular two-day block resamples.
- `data/chengdu_daily_metrics.csv` — Chengdu date-level outcomes used to reconstruct paired effects and circular two-day block intervals.
- `verify_core.py` and `requirements.txt` — a small numerical check of the headline contrasts and 95% intervals.

## Check the results

Use Python 3.10+:

```bash
python -m pip install -r requirements.txt
python verify_core.py
```

The verifier uses the 14 test dates as the resampling level, with 2,000 two-day moving-block draws. It checks arithmetic and rebuilds the paper's main intervals from the released daily aggregates. The 117,600 Beijing and 52,500 Chengdu OD/departure observations per configuration are repeated observations, not independent traffic days. Only three formal C2−C0 comparisons are reported as primary results. The Beijing E168-CDF branch is supplemental and is never pooled with the formal three-seed total.

The `departure_times` field in `dataset_scope.csv` is the **total over 14 test days**: 784 in Beijing (56 per day) and 350 in Chengdu (25 per day). Thus `od_pairs × departure_times` gives the repeated observation count; the date count must not be multiplied again.

## Data sources and scope

- Beijing Q-Traffic v2: [author project](https://github.com/JingqingZ/BaiduTraffic), described by [Liao et al. (2018)](https://doi.org/10.1145/3219819.3219895). The upstream repository does not clearly establish a separate raw-data redistribution license, so the raw archive is not included here.
- Chengdu urban link speeds: [Figshare v4 dataset](https://doi.org/10.6084/m9.figshare.7140209.v4), described by [Guo et al. (2019)](https://doi.org/10.1038/s41597-019-0060-3). The DataCite record identifies CC0 1.0; the approximately 2.1 GB raw archive is not included.

Both released datasets contain processed link speeds. Upstream completion and direction approximations cannot be fully reconstructed from them. The package supports checking the published comparisons from frozen aggregates; rebuilding path candidates, forecast models, and scenario arrays from raw records requires the separate full research pipeline and the original data.

## Provenance

The CSV and NPZ files are byte-for-byte copies of the V2 frozen evidence under the local study project. No results were refitted or selected for this GitHub release. File hashes are recorded in `SHA256SUMS.txt`.
