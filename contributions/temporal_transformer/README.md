# One-month-ahead water storage forecasts with temporal self-attention

## Experiment

At each land cell, predict next month's GRACE terrestrial water storage anomaly in cm using twelve consecutive observed months. One shared model learns across cells. The first run uses GRACE alone. The second adds monthly IMERG precipitation and MERRA-2 air temperature to each historical token. The two runs share a seed, calendar features, training budget, validation sample and evaluation observations.

The target is the aligned file's `tws` in cm, matching the existing Ridge evaluation. The separate `tws_anom` variable removes an additional monthly climatology. The team identifies its units as cm, although its units attribute is missing. The experiment leaves the source file intact. Historical observations are assumed available through the prior calendar month. Product release delays are outside this experiment.

Each token contains standardized observations plus sine and cosine of calendar month. A learned position embedding represents the lag. The latest token queries all twelve tokens through four-head self-attention. Residual connections, normalization and a feed-forward layer produce a correction to the previous TWS observation. No spatial coordinates enter the network.

## Training and evaluation

- Training targets fall in 2019–2022. Observations from 2018 may supply history. Missing calendar months are restored as missing values. Training and validation examples require a finite target and a complete history. The complete-history rule retained 1,813,383 training cell-months. Each epoch draws 32,768 of them without replacement.
- Feature means and scales use training-period observations. Twelve epochs use AdamW and standardized squared error. Checkpoints are selected using the lowest RMSE on a fixed random sample of 16,384 validation cell-months in 2023. Each selected model goes directly to testing. The archived Ridge baseline previously selected its penalty on 2023 and refitted through 2023.
- Testing uses January 2024 through September 2025. Each successive prediction can use the preceding observed test months. This measures rolling one-month forecasts. The source mask limits coverage to retained land between 60°S and 60°N.
- The evaluator intersects all finite predictions with reference observations and the supplied evaluation mask. It aligns by month and geographic coordinates. The headline score averages each cell's temporal RMSE using cosine-of-latitude weights. Pooled RMSE takes one square root after averaging squared errors across all valid cell-months with area weights. Temporal Pearson correlation requires at least three observations and variation in both series. Coverage is exported by month.

## Findings

| Model | Per-cell RMSE (cm) | MAE (cm) | Pooled RMSE (cm) | Mean temporal correlation |
|---|---:|---:|---:|---:|
| Seasonal persistence | 2.942 | 2.336 | 3.569 | 0.795 |
| Transformer with GRACE, precipitation and temperature | 3.029 | 2.415 | 3.674 | 0.791 |
| Transformer with GRACE alone | 3.169 | 2.526 | 3.899 | 0.778 |
| Ridge | 3.670 | 3.060 | 4.709 | 0.809 |
| Persistence | 4.228 | 3.464 | 5.760 | 0.740 |
| Monthly climatology | 6.707 | 5.928 | 9.124 | 0.597 |

All rows cover 976,437 common cell-months. Adding historical weather reduced per-cell RMSE by 0.140 cm (4.4%) relative to the GRACE-only run. Seasonal persistence, which adjusts the previous observation by the expected seasonal change, retained a 0.087 cm advantage over the combined Transformer. Ridge had the highest temporal correlation while making larger absolute errors than either Transformer. Results describe one seed and one test period on previously seen locations. The comparison does not isolate the individual contributions of precipitation and temperature or establish performance in unseen regions.

## Reproduction

Run from the temporal_transformer directory. The sibling combined_dataset_analysis_ridge directory provides the established scoring functions. The baseline prediction file is generated locally. For a fresh clone, run step 6 of the [project setup](../../README.md) and pass its output path to the comparison script: `sh compare.sh ../combined_dataset_analysis_ridge/rerun_results/test_predictions.nc`. The default command below uses the archived baseline file when available. Supply the team's aligned NetCDF for training. The saved predictions are sufficient for rescoring.

```sh
python -m pip install -r requirements.txt
python train.py ~/Downloads/aligned_0.5deg.nc results/grace
python train.py ~/Downloads/aligned_0.5deg.nc results/combined --combined
sh compare.sh
python plot_results.py
python -m unittest test_evaluate
```

Each run saves weights, scaler values, variable order, selected epoch, learning curve and gridded forecasts. To compare another model, add a repeated `--model NAME FILE VARIABLE` to the evaluator command in compare.sh. Supply predictions in cm on matching grid coordinates. Gridded NetCDF uses dimensions time, lat and lon. Long CSV uses one row per time, lat and lon with a named prediction column. Use the same reference file for the entire comparison and inspect coverage.csv whenever adding a model. The LSTM, TCN and ConvLSTM comparison requires their prediction files.
