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

def plot_con(Cij):

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
        left=0.1, right=0.9, bottom=0.1, top=0.9,
        wspace=0.15, hspace=0.15)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_xy = fig.add_subplot(gs[0, 1])
    
    ax.imshow(Cij, cmap='jet', aspect=1)
    ax.set_xlabel('Presynaptic')
    ax.set_ylabel('Postsynaptic')
    
    Kj = np.sum(Cij, axis=0)  # sum over pres 
    ax_histx.plot(Kj)
    ax_histx.set_xticklabels([])
    ax_histx.set_ylabel('$K_j$')

    Ki = np.sum(Cij, axis=1)  # sum over pres 
    ax_histy.plot(Ki, np.arange(0, Ki.shape[0], 1))
    ax_histy.set_yticklabels([]) 
    ax_histy.set_xlabel('$K_i$')

    con_profile(Cij, ax=ax_xy)
    
def con_profile(Cij, ax=None):
    diags = []
    for i in range(int(Cij.shape[0]/2)):
        diags.append(np.trace(Cij, offset=i))
    
    diags = np.array(diags)
    if ax is None:
        plt.plot(diags)
    else:
        ax.plot(diags)
        ax.set_xticklabels([])
        ax.set_yticklabels([])  
   
    plt.xlabel('Neuron #')
    plt.ylabel('K')
    
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

def spatial_profile(df, window=10):
    mean_df = df.groupby('neurons').mean()
    array = mean_df[['rates']].to_numpy()
        
    print(array.shape)

    smooth = circcvl(array[:, 0], windowSize=window)

    plt.plot(smooth)
    plt.xlabel('Neuron #')
    plt.ylabel('Rate (Hz)')
