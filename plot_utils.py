import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from yaml import safe_load
from scipy.stats import f_oneway
from joblib import Parallel, delayed
import scipy.stats as stat

from my_bootstrap import my_boots_ci

from decode import circcvl, decode_bump
sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**.5 - 1) / 2
width = 7.5
matplotlib.rcParams['figure.figsize'] = [width, golden_ratio * width ]

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1], 'k', '000']

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def get_lif():
    rates = np.loadtxt('../lif_cpp/new/simul/rates.txt')
    volts = np.loadtxt('../lif_cpp/new/simul/volts.txt')
    inputsE = np.loadtxt('../lif_cpp/new/simul/inputsE.txt')
    inputsI = np.loadtxt('../lif_cpp/new/simul/inputsI.txt')

    # x_stp = np.loadtxt('../lif_cpp/new/simul/x_stp.txt')
    # u_stp = np.loadtxt('../lif_cpp/new/simul/u_stp.txt')

    time = np.arange(rates.shape[0])

    X = np.stack((rates, volts, inputsE, inputsI), axis=-1)
    print(X.shape)

    variables = ['rates', 'volts', 'h_E', 'h_I']

    df = pd.DataFrame()
    idx = np.arange(0, X.shape[1], 1)
    for i_time in range(X.shape[0]):
        df_i = pd.DataFrame(X[i_time], columns=variables)
        df_i['neurons'] = idx
        df_i['time'] = time[i_time]

        # print(df_i)
        df = pd.concat((df, df_i))

    df_E = df[df.neurons<10000]

    return df_E


def get_df(filename, configname='config_bump'):
    config = safe_load(open(configname + '.yml', "r"))
    const = Bunch(config)

    Na = []
    for i_pop in range(const.N_POP):
        Na.append(int(const.N * const.frac[i_pop]))

    # print(Na)
    df = pd.read_hdf('./simul/' + filename + '.h5', mode='r')
    df = df[df.time<=3.5]

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

    # ax = sns.heatmap(pt, cmap='jet', vmax=vmax, lw=0)



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

def pev_anova(data):
    neuron_variances = np.var(data[:, :50], axis=0, ddof=1)

    # Perform one-way ANOVA to obtain F-values and p-values for each neuron
    f_values, p_values = [], []
    for i in range(50):
        # rnd = np.random.randint(1000)

        # rng = np.random.default_rng()
        # rng.shuffle(data, axis=1)

        neuron_data = data[:, i]
        other_neurons_data = np.delete(data[:,:50], i, axis=1)
        f, p = f_oneway(neuron_data, *other_neurons_data.T)
        f_values.append(f)
        p_values.append(p)

    # Calculate the percentage explained variance for each neuron
    pev = np.multiply(f_values, neuron_variances) / np.sum(neuron_variances)

    return pev

def bump_pev(filename, config, n_sim=250, ipal=0):

    name = filename

    data = []

    for i_simul in range(n_sim):

        try:
            df, df_E, df_I = get_df(name + "_id_%d" % (i_simul), config + '.yml')

            times = df_E.time.unique()
            n_time = len(times)
            n_neurons = len(df_E.neurons.unique())

            array = df_E.rates.to_numpy().reshape((n_time, n_neurons))
            data.append(array.T)

        except:
            print('error')
            data.append(np.nan)

    data = np.array(data)
    print(data.shape)

    # data = data.reshape(10, 10, n_neurons, n_time)

    for i in range(data.shape[0]):
        rnd = np.random.randint(1000)
        data[i] = np.roll(data[i], rnd, axis=0)

    # Calculate the variance for each neuron across all trials

    rng = np.random.default_rng()
    pev_list = []
    for i in range(data.shape[-1]): # time
        arr = data[:,:,i]
        pev= pev_anova(arr)
        pev_list.append(np.array(pev))

    pev_list = np.array(pev_list)
    plt.figure('pev_time')
    # plt.plot(times, pev_list, alpha=.25)
    plt.plot(times, np.mean(pev_list, 1), color=pal[ipal], alpha=1)

    plt.ylabel('PEV')
    plt.xlabel('Time (s)')


def line_phase(df):

    # data = np.loadtxt('../lif_cpp/new/simul/rates.txt')
    # times = np.arange(1, data.shape[0])
    # n_times = data.shape[0] - 1
    # n_neurons = int(data.shape[1] * 0.75)
    # array = data[1:, :int(data.shape[1]*0.75)]

    times = df.time.unique()
    n_times = len(times)
    n_neurons = len(df.neurons.unique())

    array = df.rates.to_numpy().reshape((n_times, n_neurons))

    print(n_times, n_neurons)
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
    ax[2].hlines(-(180-.25*180-180), 2.5, 4.5, color='k', ls='--')

def decode_loop(name, config, i_simul):
    try:
        df, df_E, df_I = get_df(name + "_id_%d" % (i_simul), config)

        # if 'far' in name:
        # df = df[df.time<=3.5]

        times = df_E.time.unique()
        n_times = len(times)
        n_neurons = len(df_E.neurons.unique())

        array = df_E.rates.to_numpy().reshape((n_times, n_neurons))
        m1, phase = decode_bump(array[-1])
        m1 = m1 / np.nanmean(array[-1])
    except:
        m1 = np.nan
        phase = np.nan

    return m1, phase

def diff_loop(name, config, i_simul):
    try:
        df, df_E, df_I = get_df(name + "_id_%d" % (i_simul), config)

        # if 'far' in name:
        # df = df[df.time<=3.5]

        times = df_E.time.unique()
        n_times = len(times)
        n_neurons = len(df_E.neurons.unique())

        array = df_E.rates.to_numpy().reshape((n_times, n_neurons))
        _, phase = decode_bump(array[-1])

    except:
        phase = np.nan

    return phase

def diff_loop_time(name, config, i_simul):
    try:
        df, df_E, df_I = get_df(name + "_id_%d" % (i_simul), config)
        times = df_E.time.unique()

        n_times = len(times)
        n_neurons = len(df_E.neurons.unique())

        array = df_E.rates.to_numpy().reshape((n_times, n_neurons))
        _, phase = decode_bump(array)

    except:
        pass

    return phase, times

def get_phase(filename, config, n_sim=250, THRESH=20, sign=1):

    name = filename
    phase_list = Parallel(n_jobs=-1)(delayed(diff_loop)(name, config, i_simul) for i_simul in range(n_sim))
    phase_list = np.array(phase_list) * 180.0 / np.pi - 180
    phase_list[np.abs(phase_list)>THRESH] = np.nan
    phase_list = phase_list - np.nanmean(phase_list)
    print('shape', phase_list.shape)

    idx = name.index('_')
    name2 = filename[:idx] + '2' + filename[idx:]

    phase_list2 = Parallel(n_jobs=-1)(delayed(diff_loop)(name2, config, i_simul) for i_simul in range(n_sim))
    phase_list2 = sign * (np.array(phase_list2) * 180.0 / np.pi - 180)
    phase_list2[np.abs(phase_list2)>THRESH] = np.nan

    phase_list2 = phase_list2 - np.nanmean(phase_list2)
    print('shape', phase_list2.shape)

    phase_list = np.hstack((phase_list, phase_list2))
    print('shape', phase_list.shape)

    return phase_list

def get_acc(filename, config, n_sim=250, THRESH=20):

    name = filename
    phase_list = Parallel(n_jobs=-1)(delayed(diff_loop)(name, config, i_simul) for i_simul in range(n_sim))
    phase_list = -(np.array(phase_list) * 180.0 / np.pi - 180)
    phase_list[np.abs(phase_list)>THRESH] = np.nan
    print('shape', phase_list.shape)

    idx = name.index('_')
    name2 = filename[:idx] + '2' + filename[idx:]
    phase_list2 = Parallel(n_jobs=-1)(delayed(diff_loop)(name2, config, i_simul) for i_simul in range(n_sim))
    phase_list2 = (np.array(phase_list2) * 180.0 / np.pi - 180)
    phase_list2[np.abs(phase_list2)>THRESH] = np.nan

    print('shape', phase_list2.shape)

    phase_list = np.hstack((phase_list, phase_list2))
    print('shape', phase_list.shape)

    return phase_list

def bump_diff(filename, config, n_sim=250, ipal=0, THRESH=20, bins='auto'):

    name = filename
    phase_list = Parallel(n_jobs=-1)(delayed(diff_loop)(name, config, i_simul) for i_simul in range(n_sim))
    phase_list = np.array(phase_list) * 180.0 / np.pi - 180
    phase_list = phase_list[np.abs(phase_list)<=THRESH]
    phase_list = phase_list - np.nanmean(phase_list)
    print('shape', phase_list.shape)

    # name2 = 'distractor2' + filename[10:]
    # phase_list2 = Parallel(n_jobs=-1)(delayed(diff_loop)(name2, config, i_simul) for i_simul in range(n_sim))
    # phase_list2 = np.array(phase_list2) * 180.0 / np.pi - 180
    # phase_list2 = phase_list2[np.abs(phase_list2)<=THRESH]
    # phase_list2 = phase_list2 - np.nanmean(phase_list2)
    # print('shape', phase_list2.shape)

    # phase_list = np.hstack((phase_list, phase_list2))
    # print('shape', phase_list.shape)

    plt.figure('diffusion_hist')
    _, bins, _ = plt.hist(phase_list, histtype='step', bins=bins, density=True, color=pal[ipal], lw=5, alpha=0.5)
    print('precision bias', np.nanstd(phase_list))

    plt.ylabel('Density')
    plt.xlabel('Bump Corrected Endpoint (°)')
    plt.xlim([-THRESH, THRESH])

    # bins = np.linspace(-THRESH, THRESH, n_sim)
    mu_, sigma_ = stat.norm.fit(phase_list[~np.isnan(phase_list)])
    fit_ = stat.norm.pdf(bins, mu_, sigma_)
    plt.plot(bins, fit_, color=pal[ipal], lw=5)

def bump_diff_off_on(filename, config, n_sim=250, ipal=0, THRESH=20, bins='auto'):

    bump_diff(filename + '_I0_12.00', config, n_sim, 0, THRESH, bins)
    bump_diff(filename + '_I0_24.00', config, n_sim, 1, THRESH, bins)

    plt.savefig(filename + '.svg', dpi=300)

def bump_acc_off_on(filename, config, n_sim=250, THRESH=20, bins='auto'):

    fig, ax = plt.subplots(num='accuracy_hist')

    bump_accuracy(filename + '_I0_14.00', config, n_sim, 0, THRESH, bins, ax)
    bump_accuracy(filename + '_I0_24.00', config, n_sim, 1, THRESH, bins, ax)

    plt.savefig(filename + '.svg', dpi=300)

def bump_abs_diff(filename, config, n_sim=250, ipal=0, THRESH=20, bins='auto'):

    name = filename
    phase_list = Parallel(n_jobs=-1)(delayed(diff_loop)(name, config, i_simul) for i_simul in range(n_sim))
    phase_list = np.array(phase_list) * 180.0 / np.pi - 180
    phase_list = phase_list[phase_list<=THRESH]

    phase_list = np.abs(phase_list - np.nanmean(phase_list))

    plt.figure('diffusion_hist')
    plt.hist(phase_list, histtype='step', bins=bins, density=True, color=pal[ipal])
    plt.vlines(np.nanmean(phase_list), 0, .1, ls='--', color=pal[ipal])
    print('precision bias', np.nanstd(phase_list))

    plt.ylabel('Density')
    plt.xlabel('Bump Corrected Endpoint (°)')

def bump_accuracy(filename, config, n_sim=250, ipal=0, THRESH=20, bins='auto', ax=None):

    name = filename
    phase_list = Parallel(n_jobs=-1)(delayed(diff_loop)(name, config, i_simul) for i_simul in range(n_sim))
    phase_list = np.array(phase_list) * 180.0 / np.pi - 180
    phase_list = phase_list[np.abs(phase_list)<=THRESH]
    print('shape', phase_list.shape)

    name2 = 'distractor2' + filename[10:]
    phase_list2 = Parallel(n_jobs=-1)(delayed(diff_loop)(name2, config, i_simul) for i_simul in range(n_sim))
    phase_list2 = -(np.array(phase_list2) * 180.0 / np.pi - 180)
    phase_list2[np.abs(phase_list2)>THRESH] = np.nan
    phase_list2 = phase_list2 - np.nanmean(phase_list2)
    print('shape', phase_list2.shape)

    phase_list = np.hstack((phase_list, phase_list2))
    print('shape', phase_list.shape)

    if ax is None:
        fig, ax = plt.subplots(num='accuracy_hist')

    _, bins, _ = plt.hist(phase_list, histtype='step', bins=bins, density=True, color=pal[ipal], lw=5, alpha=0.5)
    mu_, sigma_ = stat.norm.fit(phase_list[~np.isnan(phase_list)])
    fit_ = stat.norm.pdf(bins, mu_, sigma_)
    plt.plot(bins, fit_, color=pal[ipal], lw=5)

    ylim = ax.get_ylim()
    plt.vlines(np.nanmean(phase_list), 0, ylim[1], ls='--', color=pal[ipal])

    plt.ylabel('Density')
    plt.xlabel('Bump Center Endpoint (°)')

def bump_diff_thresh(filename, config, n_sim=250, ipal=0):

    name = filename
    phase_list = Parallel(n_jobs=-1)(delayed(diff_loop)(name, config, i_simul) for i_simul in range(n_sim))

    phase_list = []
    m1_list = []
    drift = np.array(phase_list)

    THRESH_LIST = np.linspace(0, 20, 20)
    drift_list = []
    per_list = []

    for thresh in THRESH_LIST:

        off = drift[np.abs(drift)<thresh]
        drift_list.append(np.nanmean(off))
        ci_off = my_boots_ci(off, np.nanmean, n_samples=1000)
        per_list.append(ci_off)

    per_list = np.array(per_list).T
    plt.plot(THRESH_LIST, drift_list, color=pal[ipal])
    plt.fill_between(THRESH_LIST, drift_list-per_list[0], drift_list+per_list[1], alpha=0.2)

    plt.xlabel('Threshold (°)')
    plt.ylabel('Shift (°)')
    plt.yticks([0, 2.5, 5.0])

def bump_diff_perf(filename, config, n_sim=250, ipal=0):

    name = filename
    phase_list = Parallel(n_jobs=-1)(delayed(diff_loop)(name, config, i_simul) for i_simul in range(n_sim))
    drift = np.array(phase_list)

    THRESH_LIST = np.linspace(0, 20, 20)
    drift_list = []
    per_list = []

    for thresh in THRESH_LIST:

        off = np.nansum(np.abs(drift)<thresh) / drift.shape[0]
        drift_list.append(off)
        ci_off = my_boots_ci(np.abs(drift)<thresh, n_samples=1000)

        per_list.append(ci_off / drift.shape[0])

    per_list = np.array(per_list).T
    plt.plot(THRESH_LIST, drift_list, color=pal[ipal])
    plt.fill_between(THRESH_LIST, drift_list-per_list[0], drift_list+per_list[1], alpha=0.2)

    plt.xlabel('Threshold (°)')
    plt.ylabel('Performance')
    plt.yticks([0, .25, .5, .75, 1])

def bump_drift(filename, config, n_sim=250, ipal=0):

    name = filename

    phase_lists = []
    cue_list = [180]
    # cue_list = [45, 90, 180, 135, 225, 270, 315]

    for cue in cue_list:
        phase_list = []

        for i_simul in range(n_sim):

            try:
                df, df_E, df_I = get_df(name + "_cue_%d_id_%d" % (cue, i_simul), config + '.yml')

                times = df_E.time.unique()
                n_times = len(times)
                n_neurons = len(df_E.neurons.unique())

                array = df_E.rates.to_numpy().reshape((n_times, n_neurons))
                m1, phase = decode_bump(array[-1])

                phase = phase * 180.0 / np.pi
                print('phase', phase, 'cue', 360-cue)
                phase_list.append(phase - (360 -cue))
            except:
                print('error')
                phase_list.append(np.nan)

        phase_lists.append(phase_list)

    phase_lists = np.hstack(np.array(phase_lists))

    plt.figure('diffusion_hist')
    plt.hist(phase_lists, histtype='step', bins='auto', density=True, color=pal[ipal])

    plt.ylabel('Density')
    plt.xlabel('End Location (°)')


def bump_diff_time(filename, config, n_sim=250):

    name = filename

    phase_list, times = zip(*Parallel(n_jobs=-1)(delayed(diff_loop_time)(name, config, i_simul) for i_simul in range(n_sim)))
    phase_list = np.array(phase_list) * 180.0 / np.pi - 180
    # phase_list = phase_list - np.nanmean(phase_list)
    # phase_list = phase_list[np.abs(phase_list)<=THRESH]
    times = np.array(times)

    plt.figure('diffusion_time')
    plt.plot(times.T, phase_list.T, alpha=.25)
    plt.fill_between([1, 1.5], y1=180, y2=-180, alpha=.2)

    plt.ylabel('Phase (°)')
    plt.xlabel('Time (s)')
    plt.yticks([-180, -90, 0, 90, 180])


def bump_drift_time(filename, config, n_sim=250):

    name = filename

    phase_list = []

    phase_list = Parallel(n_jobs=-1)(delayed(diff_loop_time)(name, config, i_simul) for i_simul in range(n_sim))
    phase_list = np.array(phase_list) * 180.0 / np.pi - 180
    # phase_list = phase_list - np.nanmean(phase_list)
    # phase_list = phase_list[np.abs(phase_list)<=THRESH]

    drift = np.sqrt(np.nanmean(phase_list.T**2, axis=1))

    plt.figure('diffusion_time')
    plt.plot(times, drift, alpha=.25)
    plt.fill_between([1, 1.5], y1=180, y2=-180, alpha=.2)

    plt.ylabel('Error (°)')
    plt.xlabel('Time (s)')
    plt.yticks([-180, -90, 0, 90, 180])

def bump_drift_cue_time(filename, config, n_sim=250):

    name = filename

    phase_lists = []
    # cue_list = [135]
    cue_list = [45, 90, 180, 135, 225, 270, 315]
    # cue_list = [45, 90, 180, 135, 225, 270, 315]

    for cue in cue_list:

        phase_list, times = zip(*Parallel(n_jobs=-1)(delayed(diff_loop_time)(name + "_cue_%d" % cue, config, i_simul) for i_simul in range(n_sim)))
        # phase_list = np.array(phase_list) * 180.0 / np.pi - 180 + cue - 180
        phase_list = np.array(phase_list) * 180.0 / np.pi - 180
        times = np.array(times)
        phase_lists.append(phase_list)

    phase_lists = np.array(phase_lists)
    times = np.array(times)
    print(phase_lists.shape)
    print(times.shape)

    plt.figure('diffusion_time')
    for i in range(len(cue_list)):
        plt.plot(times.T, phase_lists[i].T, alpha=.25, lw=4)

    plt.fill_between([1, 1.5], y1=180, y2=-180, alpha=.1)
    plt.ylabel('Phase (°)')
    plt.xlabel('Time (s)')
    plt.yticks([-180, -90, 0, 90, 180])

    plt.savefig(name + '_drift_time.svg', dpi=300)


def bump_drift_cue(filename, config, n_sim=250, ipal=0, bins='auto', cue=[], THRESH=20):

    name = filename

    phase_lists = []
    cue_list = cue
    # cue_list = [45, 90, 180, 135, 225, 270, 315, 0]
    # cue_list = [45, 90, 180, 135, 225, 270, 315]

    # for cue in cue_list:

    phase_list = Parallel(n_jobs=-1)(delayed(diff_loop)(name + "_cue_%d" % cue[0], config, i_simul) for i_simul in range(n_sim))
    # phase_list = np.array(phase_list) * 180.0 / np.pi - 180 + cue - 180
    phase_list = np.array(phase_list) * 180.0 / np.pi
    phase_list = phase_list[np.abs(phase_list)<=THRESH] #
    phase_list = phase_list - np.nanmean(phase_list)

    print(np.nanmean(phase_list))

    # phase_lists.append(phase_list)

    # phase_lists = np.array(phase_lists)
    # print(phase_lists.shape)

    # phase_list = phase_lists[0]
    plt.figure('diffusion_hist')
    _, bins, _ = plt.hist(phase_list, histtype='step', bins=bins, density=True)

    plt.ylabel('Density')
    plt.xlabel('Corrected Bump Endpoint (°)')

    mu_, sigma_ = stat.norm.fit(phase_list)
    fit_ = stat.norm.pdf(bins, mu_, sigma_)
    plt.plot(bins, fit_, color=pal[ipal], lw=5)

    # plt.xticks([-180, -90, 0, 90, 180])

    plt.savefig(name + '_drift_cue.svg', dpi=300)

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
    gain_list = [.25, .375, .5, .625, .75,  .875, 1.0]
    # gain_list = [.25, .375, .5, .625, .75,  .875, 1.0, 1.25, 1.5, 1.75]
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

def bump_I0(filename, config, n_sim=250):

    name = filename

    var_list = []
    M1_list = []

    I0_list = np.arange(12, 28, 2)

    phase_ci = []
    m1_ci = []

    for I0 in I0_list:

        phase_list = []
        m1_list = []

        m1_list, phase_list = zip(*Parallel(n_jobs=-1)(delayed(decode_loop)(name + "_I0_%.2f" % I0, config, i_simul) for i_simul in range(n_sim)))
        phase_list = np.array(phase_list) * 180.0 / np.pi / 2.5

        # print(len(phase_list), np.nanvar(phase_list))
        print(len(m1_list), np.nanmean(m1_list), len(phase_list), np.nanvar(phase_list))

        var_list.append(np.nanvar(phase_list))
        M1_list.append(np.nanmean(m1_list))

        cim = my_boots_ci(m1_list, np.nanmean, n_samples=1000)
        cip = my_boots_ci(phase_list, np.nanvar, n_samples=1000)

        # print(len(cim), len(cip))

        m1_ci.append(cim)
        phase_ci.append(cip)

    var_list = np.array(var_list)
    phase_ci = np.array(phase_ci).T

    M1_list = np.array(M1_list)
    m1_ci = np.array(m1_ci).T

    print(phase_ci.shape)
    plt.figure('m1_I0')
    plt.plot(I0_list, M1_list)
    plt.ylabel('Rel. Bump Amplitude')
    plt.xlabel('FF Input (a.u.)')
    plt.ylim([0.5, 1])
    plt.yticks([0.5, 0.75, 1])
    plt.savefig(filename + '_m1_I0.svg', dpi=300)

    fig, ax1 = plt.subplots()
    ax1.plot(I0_list, var_list, color='black')
    ax1.set_ylabel('Diffusivity ($deg^2$/$s$)', color='black')
    ax1.set_xlabel('FF Input (a.u.)', color='black')
    ax1.fill_between(I0_list, var_list - phase_ci[0], var_list + phase_ci[1], alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(I0_list, M1_list, color='grey')
    ax2.set_ylabel('Rel. Bump Amplitude', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    ax2.set_ylim([0.5, 1])
    ax2.set_yticks([0.5, 0.75, 1])
    ax2.fill_between(I0_list, M1_list-m1_ci[0], M1_list + m1_ci[1], alpha=0.2)

    plt.savefig(filename + '_all_I0.svg', dpi=300)
