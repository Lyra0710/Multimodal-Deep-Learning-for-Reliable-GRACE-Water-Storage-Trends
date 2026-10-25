# Revanth: combined dataset analysis and Ridge baseline

**Assigned scope:** combined-data correlations and distributions, plus Ridge regression.
**Status:** implementation and synthetic checks; real-data execution is pending the combined dataset.
No Ridge scores in this contribution should be presented as experimental results until the real-data run completes.

## What this contributes

Rainfall is water coming in, temperature is one influence on evaporation and snowmelt, and GRACE measures changes in total stored water. We put the three measurements on the same monthly calendar and locations. The distribution plots show what values are typical and whether later years look different. Correlations describe whether variables move together; they do not establish cause.

Ridge regression learns a weighted recipe for next month's water storage. It penalizes large coefficients to limit instability when inputs overlap. Here, a separate recipe is fitted at each grid cell. This is a temporal baseline for known locations, not a test of prediction in unseen regions.

## Required handoff

Obtain `aligned_0.5deg.nc` from the output of Rachan's `Data_Analysis_P_T_TWS_Baselines.ipynb` (provided separately in the team ZIP). That notebook prints "Saved aligned dataset for the team." The ZIP contains the notebook and Sneha's trend outputs, but **does not contain this combined dataset**. The three GRACE files already in GitHub do not include rainfall or temperature and cannot replace it.

Required variables and dimensions:

| Variable | Dimensions | Units / meaning |
|---|---|---|
| `tws` | time, lat, lon | cm, GRACE water-storage anomaly |
| `precip` | time, lat, lon | mm/month, monthly precipitation total |
| `t2m` | time, lat, lon | degC or °C, monthly temperature |
| `mask` | lat, lon | positive for retained land cells |

Use the upstream file with its history, ideally starting before January 2018, so the first 2019 targets have 12 prior rainfall months. Missing dates are restored as NaNs; no values are invented or interpolated. Duplicate months and incorrect/missing units are rejected for correction upstream. Existing anomaly variables in the handoff are ignored: this implementation calculates its own training-only monthly climatologies for descriptive correlations.

## Run locally

From the repository root, with Python 3.10 or newer:

```bash
python -m pip install -r requirements-revanth.txt
python scripts/combined_ridge.py \
  --input /path/to/aligned_0.5deg.nc \
  --config configs/grace_fo_config.yaml \
  --output-dir results/revanth_combined_ridge
python -m unittest discover -s tests -p 'test_combined_ridge.py' -v
```

The companion `notebooks/Revanth_Combined_Analysis_Ridge.ipynb` runs the same script and displays its tables and plots. It contains no invented outputs.

## Experimental choices

- Shared configuration: training target months 2019–2022, validation 2023, test 2024–2025. Missing late test months are explicitly recorded, not silently treated as a full two-year test.
- Predictors at target month t: TWS(t−1), precipitation(t−1), total precipitation over the previous 3 and 12 months, temperature(t−1), sine/cosine of calendar month, and elapsed years.
- This is retrospective one-step forecasting with observed preceding months. It does **not** simulate the publication delays of NASA products, nor forecast an entire two-year sequence without updated observations.
- Every cell gets a StandardScaler followed by Ridge. Scaling and regression are fitted only on training rows during selection. Rows with a missing target or predictor are excluded; at least 24 complete fitting rows are required.
- One global alpha from 0.1, 1, 10, 100, and 1000 is selected by pooled area-weighted validation RMSE. Final per-cell models are then refitted on training plus validation; test values never select alpha or fit the scaler/coefficients.
- Persistence, seasonal persistence, and training-plus-validation monthly climatology are computed afresh. All four models are scored on one identical test-month/cell mask. Previous observed test-month TWS may be used for the next one-step prediction.
- MAE and RMSE pool cell-month errors with cos(latitude) weights, matching the formulas in the repository's persistence script. Pearson correlation is calculated through time within each cell and then averaged with area weights. Undefined correlations are omitted, not replaced by zero.
- Correlation tables/maps use 2019–2022 only, showing raw values and values with each cell's training monthly seasonal cycle removed. Long-term trends are not removed; these are descriptive correlations without p-values.
- Distribution summaries use area-weighted means, standard deviations, quantiles, and histograms for each chronological split. Later-period distributions are diagnostic displays only, not inputs to model selection.

## Outputs

`results/revanth_combined_ridge/` will contain:

- `test_metrics.csv`: Ridge and three reference models, with evaluation counts.
- `alpha_selection.csv`: 2023 validation RMSE for each candidate alpha.
- `distributions.csv`, `histograms.csv`, `correlations.csv`: descriptive combined-data EDA.
- `training_correlations.nc`: cell-wise training correlation maps.
- `test_predictions.nc`: actual TWS, forecasts, and exact common evaluation mask on the original grid.
- `ridge_model.npz`: coefficients, intercepts, scaling parameters, feature names, alpha, and cell indices. Reconstruct with `((X - scaler_mean) / scaler_scale) @ coefficient + intercept` per cell.
- `split_manifest.csv`: target-month membership and evaluation coverage.
- `run_manifest.json`: input SHA-256, configuration, selected alpha, actual test coverage, source attributes, and limitations.
- `figures/`: distribution curves, descriptive correlations, and observed/predicted area-average time series. Area-average curves may hide local errors; use the metric table and prediction grid as well.

Large model/prediction files are ignored by Git. Publish selected CSVs, manifest, and figures after reviewing a real-data run; archive the input and large outputs in team storage with the input checksum. Retain NASA product versions/access dates in the upstream handoff metadata.

## Interpretation and remaining work

The uploaded statistical notebook's AR scores belong to Sneha's baseline. The driver notebook's stored LR/ARX scores belong to that notebook's experiment. Neither is a newly executed result for this Ridge implementation. Their masks, end dates, and predictor availability also differ; do not copy their values into a new comparison table.

The driver notebook's saved run ends in September 2025 and uses same-month climate inputs in its default nowcast mode. This contribution uses prior-month inputs. Compare those experiments only after agreeing on the forecast/nowcast task, months, masks, and metrics. The manifest exposes the actual coverage.

GRACE's 0.5° display grid oversamples its much coarser effective resolution; neighboring cells are not independent observations. This baseline also retains the supplied mask, which was selected using full-record data coverage. The time-based fitting avoids target leakage, but that upstream spatial selection is not a strictly prospective selection process. Seasonal and long-term patterns can contribute to correlations. Model coefficients are not causal effects.

Next steps: obtain the aligned handoff, run this notebook, inspect coverage and predictions, and add measured findings to the intermediate report. The individual 13-paper literature review is a separate report requirement and is not completed by this code contribution.
