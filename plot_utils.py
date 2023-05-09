import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from yaml import safe_load

from decode import circcvl, decode_bump
sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**.5 - 1) / 2
width = 7.5
matplotlib.rcParams['figure.figsize'] = [width, width * golden_ratio ]


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def get_df(filename, configname='config.yml'):
    config = safe_load(open(configname, "r"))
    const = Bunch(config)

    Na = []
    for i_pop in range(const.N_POP):
        Na.append(int(const.N * const.frac[i_pop]))

    # print(Na)
    df = pd.read_hdf('./simul/' + filename + '.h5', mode='r')

    df_E = df[df.neurons<Na[0]]
    df_EE = []
    df_I = []

    if const.N_POP==2:
        df_I = df[df.neurons>=Na[0]]
        return df, df_E, df_I

    elif const.N_POP==3:
        df_EE = df[df.neurons>=Na[0]]
        df_EE = df_EE[df_EE.neurons<Na[0]+Na[1]]
        df_I = df[df.neurons>=Na[0]+Na[1]]
        return df, df_E, df_EE, df_I
    else:
        return df, df_E, df_I

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
        diags.append(np.trace(Cij, offset=i) / Cij.shape[0])

    diags = np.array(diags)
    if ax is None:
        plt.plot(diags)
    else:
        ax.plot(diags)
        ax.set_xticklabels([])
        # ax.set_yticklabels([])

    plt.xlabel('Neuron #')
    plt.ylabel('$P_{ij}$')


def line_rates(df):
    count=0
    while count < 10:
        idx = np.random.randint(10000)
        small_df = df[df.neurons==idx]
        sns.lineplot(data=small_df, x='time', y='rates', hue='neurons', legend=None)
        count+=1

    plt.xlabel('Rates (Hz)')

def line_stp(df):
    count=0
    while count < 100:
        idx = np.random.randint(10000)
        small_df = df[df.neurons==idx]
        sns.lineplot(data=small_df, x='time', y='A_stp', hue='neurons', legend=None)
        count+=1

def line_inputs(df):
    df1 = df[df.time>.8]
    idx = np.random.randint(10000)
    small_df = df1[df.neurons==idx]
    sns.lineplot(data=small_df, x='time', y='h_E', hue='neurons', legend=None, color='r')
    sns.lineplot(data=small_df, x='time', y='h_I', hue='neurons', legend=None, color='b')
    plt.xlabel('Inputs')


def hist_rates(df, window):

    df1 = df[df.time < window[1]]
    df1 = df1[df1.time >= window[0]]

    mean_df = df1.groupby('neurons').mean()
    sns.histplot(mean_df, x=mean_df.rates, kde=True)
    plt.xlabel('Rates (Hz)')

def hist_stp(df, window=[0,1]):

    df1 = df[df.time < window[1]]
    df1 = df1[df1.time >= window[0]]

    mean_df = df1.groupby('neurons').mean()
    sns.histplot(mean_df, x=mean_df.A_stp, kde=True)
    plt.xlabel('A_stp')

def corr_rates(df):
    mean_df = df.groupby('neurons').mean()
    rates = mean_df.rates
    rij = np.outer(rates, rates)
    Cij = np.corrcoef(rij)
    plt.plot(np.mean(Cij, 1))

def hist_inputs(df):
    df1 = df[df.time>.0]
    mean_df = df1.groupby('neurons').mean()
    fig, ax = plt.subplots()

    df_E = mean_df['ff'] + mean_df['h_E']
    sns.histplot(df_E, x=df_E, kde=True, color='r', ax=ax)

    # sns.histplot(mean_df, x='h_E', kde=True, color='r', ax=ax)
    sns.histplot(mean_df, x='h_I', kde=True, color='b', ax=ax)

    df_net = mean_df['ff'] + mean_df['h_E'] + mean_df['h_I']
    sns.histplot(df_net, x=df_net, kde=True, color='k', ax=ax)

    plt.xlabel('Inputs')


def heatmap(df, vmax=20):
    df1 = df[['time','neurons','rates']]

    print(df1.head())
    pt = pd.pivot_table(df1, values ='rates',index=['neurons'],columns='time')

    n_ticks = 10
    xticks = []
    yticks = []
    # xticks = np.linspace(0, len(df1.time), n_ticks)
    # yticks = np.linspace(0, len(df1.neurons), n_ticks)

    # width=7
    # fig, ax = plt.subplots(2, 1, figsize=[1.5*width, width * golden_ratio],
    #                        gridspec_kw={'height_ratios': [1, 3]})
    # plt.tight_layout()

    # ax[0].fill_between([2/6.0, 2.5/6.0], y1=0, y2=1, alpha=.2)
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])

    ax = sns.heatmap(pt, cmap='jet', vmax=vmax, xticklabels=xticks, yticklabels=yticks, lw=0)



def spatial_profile(df, window=[0, 1], n=250, IF_NORM=0):
    df1 = df[df.time < window[1]]
    df1 = df1[df1.time >= window[0]]

    mean_df = df1.groupby('neurons').mean()
    array = mean_df[['rates']].to_numpy()

    # m1, phase = decode_bump(array)

    fig = plt.figure('spatial_profile')

    smooth = circcvl(array[:, 0], windowSize=n)
    m1, phase = decode_bump(smooth)

    print(smooth.shape)
    smooth = np.roll(smooth, int((phase / 2.0 / np.pi - 0.5 ) * smooth.shape[0]))

    if IF_NORM:
        smooth /= np.mean(array)

    theta = np.linspace(-180, 180, smooth.shape[0])

    plt.plot(theta, smooth)
    plt.xlabel('Prefered Location (°)')
    plt.ylabel('Rate (Hz)')

    plt.xticks([-180, -90, 0, 90, 180])


def spatial_profile_stp(df, window=[0,1], n=10):
    df1 = df[df.time < window[1]]
    df1 = df1[df1.time >= window[0]]

    mean_df = df1.groupby('neurons').mean()
    array = mean_df[['A_stp']].to_numpy()

    m1, phase = decode_bump(array)

    smooth = circcvl(array[:, 0], windowSize=n)
    print(smooth.shape)
    smooth = np.roll(smooth, int((phase[-1]/np.pi - 0.5 ) * smooth.shape[0]))

    plt.plot(smooth)
    plt.xlabel('Neuron #')
    plt.ylabel('A_stp')


def init(frames, ax):
    line, = ax.plot(frames[0])
    ax.set_xlabel('Neuron #')
    ax.set_ylabel('Rate (Hz)')
    ax.set_ylim([0, int(np.amax(frames)) + 1])

    return line

def animate(frame, frames, line):
    line.set_ydata(frames[frame])


def animated_bump(df, window=15, interval=10):

    frames = []
    n_frames = len(df.time.unique())
    times = df.time.unique()
    for frame in range(n_frames):
        df_i = df[df.time==times[frame]].rates.to_numpy()
        smooth = circcvl(df_i, windowSize=window)
        frames.append(smooth)

    fig, ax = plt.subplots()
    line = init(frames, ax)

    anim = FuncAnimation(fig,
        lambda i: animate(i, frames, line),
        frames=n_frames,
        interval=interval,
        repeat=False,
        cache_frame_data=False)

    plt.draw()
    # plt.show()

    writergif = PillowWriter(fps=n_frames)
    anim.save('bump.gif', writer=writergif, dpi=150)

    plt.close('all')

def line_phase(df):

    times = df.time.unique()
    n_times = len(times)
    n_neurons = len(df.neurons.unique())

    print(n_times, n_neurons)

    array = df.rates.to_numpy().reshape((n_times, n_neurons))

    print(array.shape)
    m1, phase = decode_bump(array)
    print(m1.shape, phase.shape)

    phase = phase * 180.0 / np.pi - 180.0
    
    width=7
    fig, ax = plt.subplots(1, 3, figsize=[3*width, width * golden_ratio])
    plt.tight_layout()

    m0 = np.nanmean(array, -1)

    ax[0].plot(times, m0)
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Population Rate (Hz)')

    ax[1].plot(times, m1)
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('$\mathcal{F}^1$')

    ax[2].plot(times, phase)
    ax[2].set_yticks([-180, -90, 0, 90, 180])
    ax[2].set_xlabel('Time (ms)')
    ax[2].set_ylabel('Phase (°)')

    ax[0].fill_between([1.0, 1.5], y1=0, y2=15, alpha=.2)
    ax[1].fill_between([1.0, 1.5], y1=0, y2=10, alpha=.2)
    ax[2].fill_between([1.0, 1.5], y1=180, y2=-180, alpha=.2)

    ax[0].fill_between([2.5, 3.0], y1=0, y2=15, alpha=.2)
    ax[1].fill_between([2.5, 3.0], y1=0, y2=10, alpha=.2)
    ax[2].fill_between([2.5, 3.0], y1=180, y2=-180, alpha=.2)

    ax[2].hlines(0, 0, 2.5, color='k', ls='--')
    ax[2].hlines(-(180-.25*180-180), 2.5, 6, color='k', ls='--')

def bump_diff(filename, config):

    name = filename

    phase_list = []

    for i_simul in range(250):

        try:
            df, df_E, df_I = get_df(name + "_%d" % (i_simul), config + '.yml')

            times = df_E.time.unique()
            n_times = len(times)
            n_neurons = len(df_E.neurons.unique())
          
            array = df_E.rates.to_numpy().reshape((n_times, n_neurons))
            m1, phase = decode_bump(array[-1])
        
            phase = phase * 180.0 / np.pi - 180.0
            phase_list.append(phase)
        except:
            phase_list.append(np.nan)

    phase_list = np.array(phase_list)
    plt.hist(phase_list, histtype='step', bins='auto', density=True)

    plt.ylabel('Density')
    plt.xlabel('Phase (°)')

def bump_diff_time(filename, config):

    name = filename

    phase_list = []

    for i_simul in range(25):

        try:
            df, df_E, df_I = get_df(name + "_%d" % (i_simul), config + '.yml')

            times = df_E.time.unique()
            n_times = len(times)
            n_neurons = len(df_E.neurons.unique())

            array = df_E.rates.to_numpy().reshape((n_times, n_neurons))
            m1, phase = decode_bump(array)

            phase = phase * 180.0 / np.pi - 180.0
            phase_list.append(phase)
        except:
            phase_list.append(np.nan)

    phase_list = np.array(phase_list)
    plt.plot(times, phase_list.T, alpha=.25)

    plt.ylabel('Density')
    plt.xlabel('Phase (°)')

def J0_J1_space(filename):

    name = filename

    m0_list = []
    m1_list = []

    for Jab in range(1, 11):
        for kappa in np.linspace(0, 1, 11):
            # m0_list.append(Jab)
            # m1_list.append(kappa)

            file_name = name + "_Jab_%.2f_kappa_%.2f" % (Jab, kappa)
            df, df_E, df_I = get_df(file_name, 'config_' + name + '.yml')

            times = df_E.time.unique()
            n_times = len(times)
            n_neurons = len(df_E.neurons.unique())

            array = df_E.rates.to_numpy().reshape((n_times, n_neurons))

            rates = np.nanmean(array, 0)
            m1, phase = decode_bump(rates)

            m0_list.append(np.nanmean(rates))
            m1_list.append(m1)

    width=7
    fig, ax = plt.subplots(1, 2, figsize=[2*width, width * golden_ratio])
    plt.tight_layout()

    m0_list = np.array(m0_list).reshape(10, 11)
    m1_list = np.array(m1_list).reshape(10, 11)

    ax[0].imshow(m0_list, cmap='jet', vmin=0, vmax=10, aspect='auto', extent=[1, 10, 0, 1], origin='lower')
    ax[0].set_xlabel('Jab')
    ax[0].set_ylabel('$\kappa$')

    ax[1].imshow(m1_list/m0_list, cmap='jet', vmin=0, vmax=2, aspect='auto', extent=[1, 10, 0, 1], origin='lower')
    ax[1].set_xlabel('Jab')
    ax[1].set_ylabel('$\kappa$')

    return m0_list, m1_list


def I0_S0_space(filename):

    name = filename

    m0_list = []
    m1_list = []

    for Iext in np.linspace(1, 20, 10):
        for var_ff in np.linspace(0, 100, 10):
            for id in range(1):
                file_name = name + "_I0_%.2f_S0_%.2f_id_%d" % (Iext, var_ff, id)
                df, df_E, df_I = get_df(file_name, 'config_' + name + '.yml')

                times = df_E.time.unique()
                n_times = len(times)
                n_neurons = len(df_E.neurons.unique())

                array = df_E.rates.to_numpy().reshape((n_times, n_neurons))

                rates = np.nanmean(array, 0)
                m1, phase = decode_bump(rates)

                m0_list.append(np.nanmean(rates))
                m1_list.append(m1)

    width=7
    fig, ax = plt.subplots(1, 2, figsize=[2*width, width * golden_ratio])
    plt.tight_layout()

    m0_list = np.array(m0_list).reshape(10, 10)
    m1_list = np.array(m1_list).reshape(10, 10)

    ax[0].imshow(m0_list, cmap='jet', vmin=0, vmax=10, aspect='auto', extent=[0, 100, 1, 20], origin='lower')
    ax[0].set_ylabel('$I_0$')
    ax[0].set_xlabel('$\sigma_0$')

    ax[1].imshow((m1_list/m0_list), cmap='jet', vmin=0, vmax=2, aspect='auto', extent=[0, 100, 1, 20], origin='lower')
    ax[1].set_ylabel('$I_0$')
    ax[1].set_xlabel('$\sigma_0$')

    return m0_list, m1_list

def I0_J0_space(filename):

    name = filename

    m0_list = []
    m1_list = []


    for I0 in np.arange(0, 22, 2):
        for J0 in np.arange(0, 22, 2):

            file_name = name + "_I0_%.2f_J0_%.2f" % (I0, J0)
            df, df_E, df_I = get_df(file_name, 'config_' + name + '.yml')

            times = df_E.time.unique()
            n_times = len(times)
            n_neurons = len(df_E.neurons.unique())

            array = df_E.rates.to_numpy().reshape((n_times, n_neurons))

            # rates = np.nanmean(array, 0)
            rates = array[-1]

            m1, phase = decode_bump(rates)

            m0_list.append(np.nanmean(rates))
            m1_list.append(m1)

    width=7
    fig, ax = plt.subplots(1, 2, figsize=[2*width, width * golden_ratio])
    plt.tight_layout()

    m0_list = np.array(m0_list).reshape(11, 11)
    m1_list = np.array(m1_list).reshape(11, 11)

    ax[0].imshow(m0_list, cmap='jet', vmin=0, vmax=10, aspect='auto', extent=[0, 20, 0, 20], origin='lower')
    ax[0].set_ylabel('$I_0$')
    ax[0].set_xlabel('$J_0$')

    ax[1].imshow((m1_list), cmap='jet', vmin=0, vmax=10, aspect='auto', extent=[0, 20, 0, 20], origin='lower')
    ax[1].set_ylabel('$I_0$')
    ax[1].set_xlabel('$J_0$')

    return m0_list, m1_list

def bump_gain(filename, config):

    name = filename

    var_list = []
    gain_list = [.25, .5, .75, 1.0, 1.25, 1.5, 1.75]
    M1_list = []

    for gain in gain_list:

        phase_list = []
        m1_list = []

        for i_simul in range(250):
            try :
                df, df_E, df_I = get_df( name + "_gain_%.2f_id_%d" % (gain, i_simul), config + '.yml')

                times = df_E.time.unique()
                n_times = len(times)
                n_neurons = len(df_E.neurons.unique())

                array = df_E.rates.to_numpy().reshape((n_times, n_neurons))
                m1, phase = decode_bump(array[-1])

                phase = phase * 180.0 / np.pi
                m1 = m1 / np.nanmean(array[-1])

                phase_list.append(phase)
                m1_list.append(m1)
            except:
                phase_list.append(np.nan)
                m1_list.append(np.nan)

        var_list.append(np.nanstd(phase_list))
        M1_list.append(np.nanmean(m1_list))

    var_list = np.array(var_list)

    plt.figure('diff_gain')
    plt.plot(gain_list, var_list)
    plt.ylabel('Diffusion (°)')
    plt.xlabel('Gain (a.u.)')

    M1_list = np.array(M1_list)

    plt.figure('m1_gain')
    plt.plot(gain_list, M1_list)
    plt.ylabel('Rel. Bump Amplitude (Hz)')
    plt.xlabel('Gain (a.u.)')
