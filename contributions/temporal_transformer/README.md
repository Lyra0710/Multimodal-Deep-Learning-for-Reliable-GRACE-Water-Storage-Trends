# One-month-ahead water storage forecasts with temporal self-attention

## Experiment

At each land cell, predict next month's GRACE terrestrial water storage anomaly in cm using twelve consecutive observed months. One shared model learns across cells. The first run uses GRACE alone. The second adds monthly IMERG precipitation and MERRA-2 air temperature to each historical token. The two runs share a seed, calendar features, training budget, validation sample and evaluation observations.

The input is the regenerated shared NetCDF, [p_t_tws_aligned_0.5deg_final.nc](https://drive.google.com/file/d/1PtI-xZWbaLKjYlG1mjAVoOs-Syh3WrR4/view). The target is `tws` in cm. The model assumes that observations through the previous month are available at forecast time. Each run records the dataset SHA-256, selected mask, model parameters, scaler values, and checkpoint.

Each token contains standardized observations plus sine and cosine of calendar month. A learned position embedding represents the lag. The latest token queries all twelve tokens through four-head self-attention. Residual connections, normalization and a feed-forward layer produce a correction to the previous TWS observation. No spatial coordinates enter the network.

## Training and evaluation

- Land eligibility comes from `training_mask` and the 60°S–60°N latitude limit. Both variants train on 2019–2022 targets, with data from 2018 available for input windows. A forecast example requires 12 complete input months and an observed target for training or scoring. Missing GRACE observations in 2018 reduce the eligible training targets to 39 months and 1,806,675 cell-months. The run skips incomplete histories without imputation.
- Feature means and scales use 2019–2022 observations. Twelve epochs use AdamW and standardized squared error. Each epoch samples 32,768 training pairs without replacement. The best checkpoint is selected by RMSE on the same fixed sample of 16,384 validation pairs from 2023.
- Test targets span January 2024 through September 2025. Each successive rolling forecast may use the preceding observed months. All compared models have predictions for all 21 months and 46,325 eligible cells.
- The evaluator aligns predictions by target month, latitude, and longitude and uses the intersection of valid observations and predictions. Headline RMSE averages the temporal RMSE at each cell using cosine-of-latitude weights. Per-cell MAE and temporal Pearson correlation use the same spatial weighting. Pooled RMSE takes one square root after area-weighted averaging of squared errors across cell-months.

## Findings

| Model | Per-cell RMSE (cm) | MAE (cm) | Pooled RMSE (cm) | Mean temporal Pearson r |
|---|---:|---:|---:|---:|
| LSTM with GRACE and weather | 2.858 | 2.266 | 3.552 | 0.820 |
| LSTM with GRACE | 2.899 | 2.286 | 3.570 | 0.809 |
| Transformer with GRACE and weather | 3.065 | 2.441 | 3.738 | 0.792 |
| Seasonal persistence | 3.083 | 2.453 | 3.755 | 0.788 |
| Transformer with GRACE | 3.188 | 2.539 | 3.940 | 0.785 |
| Persistence | 4.249 | 3.482 | 5.778 | 0.742 |
| Monthly climatology | 7.475 | 6.696 | 10.547 | 0.590 |

All rows cover 972,825 common cell-months. Adding precipitation and temperature together reduced Transformer per-cell RMSE by 0.123 cm (3.9%). The combined Transformer scored 0.018 cm below seasonal persistence. That margin is descriptive for this seed and test period. The LSTM rows are five-model ensemble forecasts exported by the team, and both Transformer rows use one fitted model each. The persistence baseline carries forward the most recent observation. Seasonal persistence adjusts it by the expected change from the previous month to the target month. The expected change and monthly climatology come from 2019–2022.

The intermediate Ridge scores used an earlier land mask and are available in the sibling contribution. A four-model comparison requires TCN and ConvLSTM forecast files with matching dates and coordinates.

## Reproduction

Run from the temporal_transformer directory. Supply the regenerated shared file with `training_mask`.

```sh
python -m pip install -r requirements.txt
data="$HOME/Downloads/p_t_tws_aligned_0.5deg_final.nc"
python train.py "$data" results/grace
python train.py "$data" results/combined --combined
sh compare.sh "$data"
python plot_results.py
python -m unittest test_evaluate
```

The archived comparison also includes both aligned [LSTM prediction files from commit cb266ac](https://github.com/Lyra0710/Multimodal-Deep-Learning-for-Reliable-GRACE-Water-Storage-Trends/tree/cb266ac/results/predictions). With those files available in the same checkout, reproduce all seven rows with the following command.

```sh
sh compare.sh "$data" \
  --model grace_lstm ../../results/predictions/lstm_aligned_grace_test_predictions.nc prediction \
  --model combined_lstm ../../results/predictions/lstm_aligned_combined_test_predictions.nc prediction
python plot_results.py
```

Each run saves a checkpoint, scaler values, variable order, the selected epoch, a learning curve, and gridded forecasts. Each NetCDF uses target-month timestamps and latitude and longitude coordinates. To add another model to the comparison, append `--model NAME FILE VARIABLE` with forecasts in cm. A long CSV can supply one row per time, lat, and lon with a named prediction column. Review `results/coverage.csv` when adding models.
