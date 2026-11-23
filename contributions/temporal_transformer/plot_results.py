"""Plot the shared test metric and the validation curves from saved results."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path(__file__).parent / 'results'
    scores = pd.read_csv(root/'metrics.csv').sort_values('area_weighted_mean_cell_rmse_cm')
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    colors = ['#176d75' if 'transformer' in name else '#808a96' for name in scores.model]
    labels = scores.model.str.replace('_', ' ').str.capitalize().str.replace('Grace', 'GRACE')
    bars = left.barh(labels, scores.area_weighted_mean_cell_rmse_cm, color=colors)
    left.bar_label(bars, fmt='%.2f', padding=3)
    left.invert_yaxis()
    left.set(xlabel='Area-weighted mean per-cell RMSE (cm)', title='One-month forecast error')
    left.margins(x=0.15)
    for name, label, color in [('grace', 'GRACE only', '#945311'), ('combined', 'GRACE + weather', '#176d75')]:
        curve = pd.read_csv(root/name/'learning_curve.csv')
        right.plot(curve.epoch, curve.validation_rmse_cm, marker='.', color=color, label=label)
    right.set(xlabel='Training epoch', ylabel='Validation sample RMSE (cm)', title='Checkpoint selection on 2023 data')
    right.legend(frameon=False)
    for axis in (left, right):
        axis.spines[['top','right']].set_visible(False)
    fig.savefig(root/'forecast_evaluation.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
