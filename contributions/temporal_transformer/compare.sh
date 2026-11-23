#!/bin/sh
set -eu
cd "$(dirname "$0")"
baseline=${1:-../combined_dataset_analysis_ridge/test_predictions.nc}
python evaluate.py "$baseline" results \
  --model grace_transformer results/grace/predictions.nc prediction \
  --model combined_transformer results/combined/predictions.nc prediction \
  --model ridge "$baseline" ridge \
  --model persistence "$baseline" persistence \
  --model seasonal_persistence "$baseline" seasonal_persistence \
  --model climatology "$baseline" climatology
