import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

from decode import circcvl
sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**.5 - 1) / 2
width = 7.5
matplotlib.rcParams['figure.figsize'] = [width, width * golden_ratio ]

def lineplot(df):
    count=0
    while count < 10:
        idx = np.random.randint(10000)        
        small_df = df[df.neurons==idx]
        sns.lineplot(data=small_df, x='time', y='rates', hue='neurons', legend=None)
        count+=1

def histogram(df):
    mean_df = df.groupby('neurons').mean()
    sns.distplot(mean_df.rates)
    
def heatmap(df):    
    df1 = df[['time','neurons','rates']] 
    pt = pd.pivot_table(df1, values ='rates',index=['neurons'],columns='time')
    sns.heatmap(pt, cmap='jet')

def spatial_profile(df):
    mean_df = df.groupby('neurons').mean()
    array = mean_df[['rates']].to_numpy()
        
    print(array.shape)

    smooth = circcvl(array[:, 0])

    plt.plot(smooth)
    plt.xlabel('Neuron #')
    plt.ylabel('Rate (Hz)')
