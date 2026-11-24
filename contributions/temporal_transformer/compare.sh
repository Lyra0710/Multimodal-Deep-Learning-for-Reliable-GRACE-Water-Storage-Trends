#!/bin/sh
set -eu
data=${1:?Provide the final aligned NetCDF path}
shift
data="$(cd "$(dirname "$data")" && pwd)/$(basename "$data")"
cd "$(dirname "$0")"
baseline=results/baseline_predictions.nc
python make_baselines.py "$data" "$baseline"
python evaluate.py "$baseline" results \
  --model grace_transformer results/grace/predictions.nc prediction \
  --model combined_transformer results/combined/predictions.nc prediction \
  --model persistence "$baseline" persistence \
  --model seasonal_persistence "$baseline" seasonal_persistence \
  --model climatology "$baseline" climatology \
  "$@"
