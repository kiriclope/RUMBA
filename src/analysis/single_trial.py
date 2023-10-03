import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.analysis.decode import decode_bump, circcvl


def load_conf(conf_name, repo_root, **kwargs):
    """Return params from conf file."""
    conf_path = repo_root + '/conf/'+ conf_name + '.yml'
    print('Loading config from', conf_path)
    param = safe_load(open(conf_path, "r"))
    
    param["FILE_NAME"] = sim_name
    param.update(kwargs)

    for k, v in param.items():
        setattr(self, k, v)

    return param


def load_data(sim_name, repo_root):
    """Return df with Fourier moments and phase."""    
    df = pd.read_hdf(repo_root + "/data/simul/" + sim_name + ".h5", mode="r")
    print(df.head())

    return df

    
def plot_raster(df, **kwargs):
    """Plot raster (neuron id vs time vs rates)."""
    pt = pd.pivot_table(df, values="rates", index=["neurons"], columns="time")

    fig, ax = plt.subplots()
    sns.heatmap(pt, cmap="jet", ax=ax, vmax=15, vmin=0)
    
    ax.set_yticks([0, 500, 1000], [0, 500, 1000])
    ax.set_xticks([0, 2, 4, 6, 8], [0, 1, 2, 3, 4])

    
def plot_rates_dist(df, **kwargs):
    """Plot rates distributions."""
    mean_df = df.groupby("neurons").mean()
    mean_df[mean_df.rates<.01] = np.nan
    
    sns.histplot(mean_df, x=mean_df.rates, kde=True, color='r')
    plt.xlabel("Rates (au)")
    plt.show()
    

def get_tuning(df):
    """Return df with Fourier moments and phase."""
    
    data = df.groupby(['time'])['rates'].apply(decode_bump).reset_index()
    data[['m0', 'm1', 'phase']] = pd.DataFrame(data['rates'].tolist(), index=data.index)
    data = data.drop(columns=['rates'])
    
    print(data.head())

    return data
    

def plot_fourier_time(df, NORM_M1=False):    
    """Plot Fourier moments and phase vs time."""
 
    golden_ratio = (5**.5 - 1) / 2
    width = 6
    height = width * golden_ratio
    
    fig, ax = plt.subplots(1, 3, figsize=[2*width, height])
  
    sns.lineplot(data=df, x='time', y='m0', legend=False, lw=2, ax=ax[0])
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('$\mathcal{F}_0 (Hz)$')
    ax[1].set_xticks([0, 1, 2, 3, 4])

    if NORM_M1:
        m1 = df['m1']/df['m0']
    else:
        m1 = df['m1']
    
    sns.lineplot(x=df['time'], y=m1, legend=False, lw=2, ax=ax[1])
    ax[1].set_xlabel('Time (s)')
    if NORM_M1:
        ax[1].set_ylabel('$\mathcal{F}_1 / \mathcal{F}_0$')
    else:
        ax[1].set_ylabel('$\mathcal{F}_1$')
    
    ax[1].set_xticks([0, 1, 2, 3, 4])

    sns.lineplot(x=df['time'], y=df['phase']*180/np.pi, legend=False, lw=2, ax=ax[2])
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('$\phi$ (°)')
    ax[2].set_xticks([0, 1, 2, 3, 4])
    ax[2].set_yticks([0, 90, 180, 270, 360])
    plt.show()

    
def plot_tuning_profile(df, WINDOW_SIZE=10, T_WINDOW=[1.5, 2]):
    """Plot tuning profile in a given window."""
    df_epoch = df[df.time < T_WINDOW[1]]
    df_epoch = df_epoch[df_epoch.time >= T_WINDOW[0]]

    mean_epoch = df_epoch.groupby("neurons").mean()
    array = mean_epoch[["rates"]].to_numpy()

    X_epoch = circcvl(array[:, 0], windowSize=WINDOW_SIZE)
    m0, m1, phase = decode_bump(X_epoch)

    X_epoch = np.roll(X_epoch, int((phase / 2.0 / np.pi - 0.5) * X_epoch.shape[0]))

    theta = np.linspace(-180, 180, X_epoch.shape[0])
    fig, ax = plt.subplots()
    ax.plot(theta, X_epoch)
    ax.set_xlabel("Prefered Location (°)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_ylim([0, 15])
    
