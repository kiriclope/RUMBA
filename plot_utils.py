import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**.5 - 1) / 2
width = 6
matplotlib.rcParams['figure.figsize'] = [width, width * golden_ratio ]

def lineplot(df):
    small_df = df[df.neurons<10]
    sns.lineplot(data=small_df, x='time', y='rates', hue='neurons', legend=None)

def histogram(df):
    mean_df = df.groupby('neurons').mean()
    sns.distplot(mean_df.rates)
    
def heatmap(df):    
    # df1 = gaussian_filter(df, sigma=.1)
    df1 = df[['time','neurons','rates']] 
    # df1['neurons'] = gaussian_filter(df1['neurons'], sigma=1)

    pt = pd.pivot_table(df1, values ='rates',index=['neurons'],columns='time')
    sns.heatmap(pt, cmap='jet')

    return pt 
