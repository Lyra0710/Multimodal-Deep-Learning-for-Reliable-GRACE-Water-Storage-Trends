"""Recreate presentation figures from the measured CSV results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
m=pd.read_csv(ROOT/'test_metrics.csv').set_index('model')
order=['seasonal_persistence','ridge','persistence','climatology']
labels=['Seasonal persistence','Ridge','Persistence','Monthly climatology']
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for ax,metric,title in zip(axes,['mae_cm','rmse_cm'],['MAE: average absolute error','RMSE: emphasizes large errors']):
 vals=m.loc[order,metric]
 bars=ax.barh(labels,vals,color=['#24866e','#3477bd','#adb5bd','#adb5bd'])
 ax.invert_yaxis(); ax.set(xlabel='cm — lower is better',title=title,xlim=(0,vals.max()*1.18))
 ax.bar_label(bars,fmt='%.2f',padding=5); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Same 21 test months • Same 46,497 cells • Area-weighted pooled errors',fontsize=12)
fig.savefig(ROOT/'figures/model_comparison.png',dpi=180);plt.close(fig)
c=pd.read_csv(ROOT/'correlations.csv')
pretty=['Water storage ↔ rainfall','Water storage ↔ temperature','Rainfall ↔ temperature']
fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
import numpy as np
pos=np.arange(3)
ax.barh(pos-.18,c.area_weighted_mean_r.iloc[:3],height=.34,label='Raw measurements',color='#3477bd')
ax.barh(pos+.18,c.area_weighted_mean_r.iloc[3:],height=.34,label='Usual seasonal cycle removed',color='#24866e')
ax.set(yticks=pos,yticklabels=pretty,xlim=(-.4,.4),xlabel='Area-weighted mean correlation within each location',title='Training data: how the measurements move together')
ax.axvline(0,color='#999999',lw=1);ax.legend(loc='lower left');ax.invert_yaxis()
fig.savefig(ROOT/'figures/correlations_readable.png',dpi=180);plt.close(fig)
