import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import bootstrap

from plot_utils import get_phase, get_acc

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]

def my_boots_ci(X, statfunc, n_samples=10000, method="BCa", alpha=0.05):

    boots_samples = bootstrap(
        (X,),
        statistic=statfunc,
        n_resamples=n_samples,
        method=method,
        confidence_level=1.0 - alpha,
    )

    # print(boots_samples)

    ci = [boots_samples.confidence_interval.low, boots_samples.confidence_interval.high]
    mean_boots = np.mean(boots_samples.bootstrap_distribution)

    ci[0] = mean_boots - ci[0]
    ci[1] = ci[1] - mean_boots

    return ci


def plot_dtheta_distance(filename, config, n_sim=500, THRESH=30):

    df_off = get_phase(filename + '_close_I0_14.00', config, n_sim, THRESH)
    df_on = get_phase(filename + '_close_I0_24.00', config, n_sim, THRESH)

    df_off2 = get_phase(filename + '_far_I0_14.00', config, n_sim, THRESH)
    df_on2 = get_phase(filename + '_far_I0_24.00', config, n_sim, THRESH)

    df_off = np.vstack((df_off, df_off2))
    df_on = np.vstack((df_on, df_on2))

    print(df_off.shape)
    std_off, std_on = [], []
    ci_off, ci_on = [], []

    dist_list = np.array([45, 90])

    for i in range(2):

        std_off.append(np.nanstd(df_off[i]))
        std_on.append(np.nanstd(df_on[i]))

        ci_off.append(my_boots_ci(df_off[i], statfunc=np.nanstd))
        ci_on.append(my_boots_ci(df_on[i], statfunc=np.nanstd))

    figname = "dtheta_distance"
    plt.figure(figname)
    plt.plot(dist_list, std_off, "-o", color=pal[0])
    plt.plot(dist_list + 5, std_on, "-o", color=pal[1])
    plt.xticks([45, 90], ['Close', 'Far'])
    plt.xlabel("Distance btw Targets (°)")
    plt.ylabel("Precision Bias (°)")

    plt.errorbar(dist_list, std_off, yerr=np.array(ci_off).T, color=pal[0])
    plt.errorbar(dist_list + 5, std_on, yerr=np.array(ci_on).T, color=pal[1])
    plt.savefig(figname + ".svg", dpi=300)

def plot_error_distance(filename, config, n_sim=500, THRESH=30):

    df_off = get_acc(filename + '_close_I0_14.00', config, n_sim, THRESH)
    df_on = get_acc(filename + '_close_I0_24.00', config, n_sim, THRESH)

    df_off2 = get_acc(filename + '_far_I0_14.00', config, n_sim, THRESH)
    df_on2 = get_acc(filename + '_far_I0_24.00', config, n_sim, THRESH)

    df_off = np.vstack((df_off, df_off2))
    df_on = np.vstack((df_on, df_on2))

    print(df_off.shape)
    std_off, std_on = [], []
    ci_off, ci_on = [], []

    dist_list = np.array([45, 90])

    for i in range(2):

        std_off.append(np.nanmean(df_off[i]))
        std_on.append(np.nanmean(df_on[i]))

        ci_off.append(my_boots_ci(df_off[i], statfunc=np.nanmean))
        ci_on.append(my_boots_ci(df_on[i], statfunc=np.nanmean))

    figname = "error_distance"
    plt.figure(figname)
    plt.plot(dist_list, std_off, "-o", color=pal[0])
    plt.plot(dist_list + 5, std_on, "-o", color=pal[1])
    plt.xticks([45, 90], ['Close', 'Far'])
    plt.xlabel("Distance btw Targets (°)")
    plt.ylabel("Distraction Bias (°)")

    plt.errorbar(dist_list, std_off, yerr=np.array(ci_off).T, color=pal[0])
    plt.errorbar(dist_list + 5, std_on, yerr=np.array(ci_on).T, color=pal[1])
    plt.savefig(figname + ".svg", dpi=300)
