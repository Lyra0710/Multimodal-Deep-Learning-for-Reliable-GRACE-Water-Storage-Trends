# Multimodal-Deep-Learning-for-Reliable-GRACE-Water-Storage-Trends
# DATA 255 — Group 5 Project

**Project:** Multimodal Deep Learning for Reliable GRACE Water Storage Trends

## Project Overview

This project investigates whether multimodal deep-learning models can reliably predict changes in terrestrial water storage (TWS) using NASA GRACE/GRACE-FO observations, precipitation, and temperature data.

We aim to compare different forecasting approaches, analyze water-storage trends, and evaluate model reliability across different environmental conditions.

### Current Scope: Intermediate Project

For the intermediate submission, we will focus on:

- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Statistical trend analysis
- Baseline model implementation and evaluation
- Literature review

Advanced deep-learning models (LSTM, TCN, Transformer), multimodal fusion experiments, uncertainty estimation, and explainability will be addressed in the final project.

---

## Folder Guidelines

### `data/`

Contains datasets used throughout the project.

- `raw/`: Original downloaded NASA datasets (GRACE/GRACE-FO, GPM IMERG, MERRA-2).
- `processed/`: Cleaned, standardized, and merged datasets ready for analysis and modeling.

**Important:** Do not commit large datasets to GitHub. Keep them locally or in shared storage. Document their sources and download instructions so other members can reproduce the preprocessing.

### `scripts/`

Contains reusable Python scripts for data preprocessing, analysis, modeling, and evaluation.

Each member should place their implementation in the relevant `.py` files.

Keep scripts organized and avoid duplicating functionality already implemented by another member.

### `notebooks/`

Contains Jupyter notebooks for exploratory analysis, experiments, and visualizations.

Each member should primarily work in their assigned notebook.

Use notebooks to demonstrate the analysis and results. Move reusable functions into `scripts/` whenever practical.

### `models/`

Contains trained model files, such as `.pkl` or `.joblib`.

Large model files should not be committed to GitHub.

### `results/`

Contains outputs generated from the experiments.

- `figures/`: EDA plots, trend maps, actual-vs-predicted plots, and other visualizations.
- `metrics/`: Model evaluation results, including MAE, RMSE, correlation, and comparison tables.

Use descriptive filenames, such as `grace_missing_months.png` or `persistence_metrics.csv`.

### `reports/`

Contains written documentation and report materials.

Each member is responsible for individually reviewing at least 13 relevant research papers for the intermediate submission. Use APA citation format and write the survey in your own words.

---
