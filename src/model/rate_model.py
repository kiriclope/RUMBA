import sys
from math import erf
from time import perf_counter

import numpy as np
from numba import jit, set_num_threads
from pandas import DataFrame, HDFStore, concat
from scipy.sparse import csc_matrix
from src.analysis.decode import decode_bump
from src.mean_field.mean_field_spec import get_mf_spec, m0_func
from src.model.connectivity import (gaussian, generate_Cab, moving_average,
                                    numba_multiple_maps, numba_normal,
                                    numba_update_Cij, numba_update_DJij,
                                    numba_update_local_field, theta_mat)
from src.model.plasticity import STP_Model, numba_markram_stp
from tqdm import tqdm
from yaml import safe_load


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def pertur_func(theta, I0, SIGMA0, PHI0, TYPE="cos"):
    if TYPE == "cos":
        res = I0 * (1.0 + SIGMA0 * np.cos(theta - PHI0))
        return res * (res > 0)
    else:
        return I0 * gaussian(theta - PHI0, SIGMA0)


def nd_numpy_to_nested(X, N_POP=2, IF_STP=0):
    """Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints)
    into pandas DataFrame (with time series as pandas Series in cells)
    Parameters
    ----------
    X : NumPy ndarray, input
    Returns
    -------
    pandas DataFrame
    """

    # print(X.shape)

    if IF_STP:
        if N_POP == 2:
            variables = ["rates", "ff", "h_E", "h_I", "u_stp", "x_stp", "A_stp"]
        else:
            variables = ["rates", "ff", "h_E", "u_stp", "x_stp", "A_stp"]
    else:
        if N_POP == 2:
            variables = ["rates", "ff", "h_E", "h_I"]
        else:
            variables = ["rates", "ff", "h_E"]

    df = DataFrame()
    idx = np.arange(0, X.shape[1], 1)
    for i_time in range(X.shape[0]):
        df_i = DataFrame(X[i_time, :, 1:], columns=variables)
        df_i["neurons"] = idx
        df_i["time"] = X[i_time, 0, 0]

        # print(df_i)
        df = concat((df, df_i))

    # print(df)

    return df


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_erf(x):
    res = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        res[i] = erf(x[i])
    return res


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def TF(x, thresh=None, tfname="TL", tfgain=0.5):
    # if tfname=='NL':
    # return 1.0 * (x >= 1.0) * np.sqrt(np.abs(4.0 * x - 3.0)) + x * x * (x > 0)
    # return np.where(x >= 1.0, np.sqrt(np.abs(4.0 * x - 3.0)), x * x * (x > 0)).astype(np.float64)
    # else:

    # return x * (x > 0.0)
    # return x * (x > 0) * (x <= thresh)  + (tfgain * x + thresh * (1.0 - tfgain)) * (x >= thresh)

    # elif tfname=='Sig':
    return thresh * (0.5 * (1.0 + numba_erf(x / np.sqrt(2.0)))).astype(np.float64)
    # elif tfname=='LIF':
    # return (- 1.0 * (x > 1.0) / np.log(1.0 - 1.0 / x)).astype(np.float64)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_ff_inputs(
    ff_inputs, ff_inputs_0, EXP_DT_TAU_FF, DT_TAU_FF, VAR_FF, FF_DYN=0
):
    if FF_DYN == 1:
        ff_inputs = ff_inputs * EXP_DT_TAU_FF[0]
        ff_inputs = ff_inputs + DT_TAU_FF[0] * ff_inputs_0
    elif FF_DYN == 2:
        ff_inputs[:] = (
            np.sqrt(VAR_FF[0]) * np.random.normal(0, 1.0, ff_inputs.shape[0])
            + ff_inputs_0
        )
    else:
        ff_inputs = ff_inputs_0

    return ff_inputs


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_inputs(Cij, rates, inputs, csumNa, EXP_DT_TAU_SYN, SYN_DYN=1):
    if SYN_DYN == 0:
        for i_pop in range(inputs.shape[0]):
            inputs[i_pop] = np.dot(
                Cij[:, csumNa[i_pop] : csumNa[i_pop + 1]],
                rates[csumNa[i_pop] : csumNa[i_pop + 1]],
            )
    else:
        for i_pop in range(inputs.shape[0]):
            inputs[i_pop] = inputs[i_pop] * EXP_DT_TAU_SYN[i_pop]
            inputs[i_pop] = inputs[i_pop] + np.dot(
                Cij[:, csumNa[i_pop] : csumNa[i_pop + 1]],
                rates[csumNa[i_pop] : csumNa[i_pop + 1]],
            )

    return inputs


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_rates(
    rates,
    inputs,
    ff_inputs,
    inputs_NMDA,
    thresh,
    TF_NAME,
    TF_GAIN,
    csumNa,
    EXP_DT_TAU_MEM,
    DT_TAU_MEM,
    RATE_DYN=0,
    IF_NMDA=0,
    RATE_NOISE=0,
    RATE_VAR=0.0,
):
    net_inputs = ff_inputs

    for i_pop in range(inputs.shape[0]):
        net_inputs = net_inputs + inputs[i_pop]

    if IF_NMDA:
        for i_pop in range(inputs.shape[0]):
            net_inputs = net_inputs + inputs_NMDA[i_pop]

    if RATE_DYN == 0:
        rates = TF(net_inputs, thresh, TF_NAME, TF_GAIN)
    else:
        for i_pop in range(inputs.shape[0]):
            rates[csumNa[i_pop] : csumNa[i_pop + 1]] = (
                rates[csumNa[i_pop] : csumNa[i_pop + 1]] * EXP_DT_TAU_MEM[i_pop]
            )
            # rates[csumNa[i_pop]:csumNa[i_pop+1]] = rates[csumNa[i_pop]:csumNa[i_pop+1]] + net_inputs[csumNa[i_pop]:csumNa[i_pop+1]] * (net_inputs[csumNa[i_pop]:csumNa[i_pop+1]] > 0) * DT_TAU_MEM[i_pop]
            rates[csumNa[i_pop] : csumNa[i_pop + 1]] = rates[
                csumNa[i_pop] : csumNa[i_pop + 1]
            ] + DT_TAU_MEM[i_pop] * TF(
                net_inputs[csumNa[i_pop] : csumNa[i_pop + 1]],
                thresh[csumNa[i_pop] : csumNa[i_pop + 1]],
                TF_NAME,
                TF_GAIN,
            )

    if RATE_NOISE:
        rates = rates[:] + np.sqrt(RATE_VAR) * np.random.normal(0, 1.0, rates.shape[0])

    return rates


class Network:
    def __init__(self, **kwargs):
        const = Bunch(kwargs)

        self.IF_LOAD_MAT = const.IF_LOAD_MAT
        self.IF_SAVE_MAT = const.IF_SAVE_MAT

        self.PHASE = const.PHASE * np.pi / 180.0

        self.SAVE = const.SAVE
        self.verbose = const.verbose
        # if self.verbose:
        #     print(kwargs.keys())

        self.FILE_NAME = const.FILE_NAME
        self.TF_NAME = const.TF_NAME
        self.TF_GAIN = const.TF_GAIN

        self.RATE_DYN = const.RATE_DYN
        self.RATE_NOISE = const.RATE_NOISE
        self.RATE_VAR = const.RATE_VAR

        self.SYN_DYN = const.SYN_DYN

        # SIMULATION
        self.DT = const.DT

        self.DURATION = float(const.DURATION)
        self.N_STEPS = int(self.DURATION / self.DT)

        self.N_STEADY = int(const.T_STEADY / self.DT)
        self.N_WINDOW = int(const.T_WINDOW / self.DT)

        self.N_STIM_ON = int(const.T_STIM_ON / self.DT)
        self.N_STIM_OFF = int(const.T_STIM_OFF / self.DT)

        self.N_CUE_ON = int((const.T_STIM_ON + 1.5) / self.DT)
        self.N_CUE_OFF = int((const.T_STIM_OFF + 1.5) / self.DT)

        # PARAMETERS
        self.N_POP = int(const.N_POP)
        self.N = int(const.N)
        self.K = const.K

        self.ones_vec = (
            np.ones((self.N,), dtype=np.float64) / self.N_STEPS * self.DURATION
        )

        self.TAU_SYN = const.TAU_SYN
        self.TAU_FF = const.TAU_FF

        self.VAR_FF = np.array(const.VAR_FF, dtype=np.float64)

        self.TAU_MEM = const.TAU_MEM
        self.TAU_THRESH = const.TAU_THRESH

        self.Na = []
        self.Ka = []

        self.EXP_DT_TAU_SYN = []
        self.DT_TAU_SYN = []

        self.IF_STP = const.IF_STP

        self.IF_NMDA = const.IF_NMDA
        if self.IF_NMDA:
            self.TAU_NMDA = const.TAU_NMDA
            self.EXP_DT_TAU_NMDA = []
            self.DT_TAU_NMDA = []

        self.FF_DYN = const.FF_DYN
        self.EXP_DT_TAU_FF = []
        self.DT_TAU_FF = []

        self.EXP_DT_TAU_MEM = []
        self.DT_TAU_MEM = []

        self.EXP_DT_TAU_THRESH = []
        self.DT_TAU_THRESH = []

        for i_pop in range(self.N_POP):
            self.Na.append(int(self.N * const.frac[i_pop]))
            # self.Ka.append(self.K * const.frac[i_pop])
            self.Ka.append(self.K)

            self.EXP_DT_TAU_SYN.append(np.exp(-self.DT / self.TAU_SYN[i_pop]))
            self.DT_TAU_SYN.append(self.DT / self.TAU_SYN[i_pop])

            if self.IF_NMDA:
                self.EXP_DT_TAU_NMDA.append(np.exp(-self.DT / self.TAU_NMDA[i_pop]))
                self.DT_TAU_NMDA.append(self.DT / self.TAU_NMDA[i_pop])

            self.EXP_DT_TAU_FF.append(np.exp(-self.DT / self.TAU_FF[i_pop]))
            self.DT_TAU_FF.append(self.DT / self.TAU_FF[i_pop])

            self.EXP_DT_TAU_MEM.append(np.exp(-self.DT / self.TAU_MEM[i_pop]))
            self.DT_TAU_MEM.append(self.DT / self.TAU_MEM[i_pop])

            self.EXP_DT_TAU_THRESH.append(np.exp(-self.DT / self.TAU_THRESH[i_pop]))
            self.DT_TAU_THRESH.append(self.DT / self.TAU_THRESH[i_pop])

        self.Na = np.array(self.Na, dtype=np.int64)
        self.Ka = np.array(self.Ka, dtype=np.float64)

        self.EXP_DT_TAU_FF = np.array(self.EXP_DT_TAU_FF, dtype=np.float64)
        self.DT_TAU_FF = np.array(self.DT_TAU_FF, dtype=np.float64)

        self.EXP_DT_TAU_SYN = np.array(self.EXP_DT_TAU_SYN, dtype=np.float64)
        self.DT_TAU_SYN = np.array(self.DT_TAU_SYN, dtype=np.float64)

        if self.IF_NMDA:
            self.EXP_DT_TAU_NMDA = np.array(self.EXP_DT_TAU_NMDA, dtype=np.float64)
            self.DT_TAU_NMDA = np.array(self.DT_TAU_NMDA, dtype=np.float64)

        self.EXP_DT_TAU_MEM = np.array(self.EXP_DT_TAU_MEM, dtype=np.float64)
        self.DT_TAU_MEM = np.array(self.DT_TAU_MEM, dtype=np.float64)

        self.EXP_DT_TAU_THRESH = np.array(self.EXP_DT_TAU_THRESH, dtype=np.float64)
        self.DT_TAU_THRESH = np.array(self.DT_TAU_THRESH, dtype=np.float64)

        self.csumNa = np.concatenate(([0], np.cumsum(self.Na)))

        self.THRESH_DYN = const.THRESH_DYN
        self.THRESH = np.array(const.THRESH, dtype=np.float64)

        self.M0 = const.M0

        self.Jab = np.array(const.Jab, dtype=np.float64).reshape(self.N_POP, self.N_POP)

        if self.verbose:
            print("Jab", self.Jab)

        self.STRUCTURE = np.array(const.STRUCTURE).reshape(self.N_POP, self.N_POP)
        self.SIGMA = np.array(const.SIGMA, dtype=np.float64).reshape(
            self.N_POP, self.N_POP
        )
        self.KAPPA = np.array(const.KAPPA, dtype=np.float64).reshape(
            self.N_POP, self.N_POP
        )

        if self.verbose:
            print("SIGMA", self.SIGMA)

        if self.verbose:
            print("KAPPA", self.KAPPA)

        self.SIGMA = self.SIGMA / np.abs(self.Jab)

        self.GAIN = const.GAIN
        self.Jab *= const.GAIN

        if self.IF_NMDA:
            self.Jab_NMDA = (
                0.1 * self.Jab[:, 0] * self.DT_TAU_NMDA[i_pop] / np.sqrt(self.Ka[0])
            )

        self.Iext = np.array(const.Iext, dtype=np.float64)

        if self.verbose:
            print("Iext", self.Iext)

        try:
            self.mf_rates = -np.dot(np.linalg.inv(self.Jab), self.Iext)
        except:
            self.mf_rates = np.nan

        if self.SYN_DYN:
            for i_pop in range(self.N_POP):
                self.Jab[:, i_pop] = (
                    self.Jab[:, i_pop]
                    * self.DT_TAU_SYN[i_pop]
                    / np.sqrt(self.Ka[i_pop])
                )
        else:
            for i_pop in range(self.N_POP):
                self.Jab[:, i_pop] = self.Jab[:, i_pop] / np.sqrt(self.Ka[i_pop])

        self.Jab[np.isinf(self.Jab)] = 0

        self.Iext *= np.sqrt(self.Ka[0]) * self.M0

        self.PERT_TYPE = const.PERT_TYPE
        self.I0 = np.array(const.I0, dtype=np.float64)
        self.I0 *= self.M0  # * np.sqrt(self.Ka[0])

        self.I1 = np.random.normal(const.I1, np.sqrt(2.0 * np.array(const.I1)))
        self.I1 *= self.M0  # * np.sqrt(self.Ka[0])

        if const.PHI0 == "None":
            self.PHI0 = np.random.uniform(2 * np.pi)
        else:
            self.PHI0 = const.PHI0 * np.pi / 180.0

        self.PHI1 = self.PHI0 - const.DPHI * np.pi

        self.SIGMA0 = const.SIGMA0
        self.SIGMA_EXT = const.SIGMA_EXT

        self.SEED = const.SEED
        if self.SEED == "None":
            self.SEED = None

        # LEARNING
        self.IF_LEARNING = const.IF_LEARNING
        self.TAU_LEARN = 250.0
        self.DT_TAU_LEARN = self.DT / self.TAU_LEARN

        # self.KAPPA_LEARN = 1.0
        self.KAPPA_LEARN = self.KAPPA[0][0] / np.sqrt(self.Ka[0])
        self.KAPPA_DT_TAU_LEARN = self.KAPPA_LEARN * self.DT_TAU_LEARN
        self.EXP_DT_TAU_LEARN = np.exp(-self.DT / self.TAU_LEARN, dtype=np.float64)

        self.ALPHA = 1.0
        # self.ALPHA = np.sqrt(self.Ka[0])
        self.ETA_DT = self.DT / self.TAU_SYN[0]

        self.rates = np.ascontiguousarray(np.zeros((self.N,), dtype=np.float64))
        self.inputs = np.zeros((self.N_POP, self.N), dtype=np.float64)
        self.inputs_NMDA = np.zeros((self.N_POP, self.N), dtype=np.float64)

        rng = np.random.default_rng()
        ## random initial conditions
        mean = [2, 2, 5]
        var = [0.5, 0.5, 1]

        for i_pop in range(self.N_POP):
            self.rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = rng.normal(
                mean[i_pop], var[i_pop], self.Na[i_pop]
            )
            self.inputs[i_pop] = rng.normal(10, 10 / 4.0, self.N)

        ## initial conditions from self consistent eqs
        # u0, u1, alpha = get_mf_spec("config")
        # for i_pop in range(self.N_POP):
        #     if i_pop==0:
        #         self.rates[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = TF(u0[i_pop] + np.sqrt(alpha[i_pop])
        #                                                             * rng.standard_normal(self.Na[i_pop], dtype=np.float64))
        #     if i_pop==1:
        #         self.rates[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = TF(u0[0] + np.sqrt(alpha[0])
        #                                                                 * rng.standard_normal(self.Na[1], dtype=np.float64))

        #     if i_pop==2:
        #         self.rates[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = TF(u0[1] + np.sqrt(alpha[1])
        #                                                                 * rng.standard_normal(self.Na[2], dtype=np.float64))

        # print(self.rates[:5])
        # self.mf_rates = m0_func(u0, u1, alpha)

        self.ff_inputs = np.zeros((self.N,), dtype=np.float64)
        self.ff_inputs_0 = np.ones((self.N,), dtype=np.float64)
        self.thresh = np.ones((self.N,), dtype=np.float64)

        # for i_pop in range(self.N_POP):
        #     theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop])

        #     self.rnd_cos = rng.standard_normal(theta.shape[0], dtype=np.float64) * np.cos(theta)
        #     self.rnd_sin = rng.standard_normal(theta.shape[0], dtype=np.float64) * np.sin(theta)

        #     self.ff_inputs_0[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = self.Iext[i_pop] * (1.0 + self.SIGMA_EXT / np.sqrt(self.Ka[i_pop])  * (self.rnd_cos + self.rnd_sin))
        #     # self.ff_inputs_0[self.csumNa[1]:self.csumNa[1+1]] = self.ff_inputs_0[self.csumNa[1]:self.csumNa[1+1]] * self.Iext[1]

        for i_pop in range(self.N_POP):
            self.ff_inputs_0[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (
                self.ff_inputs_0[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                * self.Iext[i_pop]
            )
            self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (
                self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                * self.THRESH[i_pop]
            )

    def print_params(self):
        print("Parameters:")
        print("N", self.N, "Na", self.Na)
        print("K", self.K, "Ka", self.Ka)

        print("Iext", self.Iext)
        print("Jab", self.Jab.flatten())

        print("MF Rates:", self.mf_rates)

    def update_inputs(self, Cij):
        if self.SYN_DYN == 0:
            for i_pop in range(self.N_POP):
                self.inputs[i_pop] = np.tensordot(
                    Cij[:, self.csumNa[i_pop] : self.csumNa[i_pop + 1]],
                    self.rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]],
                    1,
                )
                # self.inputs[i_pop] = Cab @ rb
            else:
                for i_pop in range(self.N_POP):
                    self.inputs[i_pop] = self.inputs[i_pop] * self.EXP_DT_TAU_SYN[i_pop]
                    Cab = Cij[:, self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                    rb = self.rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                    self.inputs[i_pop] = self.inputs[i_pop] + np.dot(Cab, rb)

    def update_rates(self):
        net_inputs = self.ff_inputs

        for i_pop in range(self.N_POP):
            net_inputs = net_inputs + self.inputs[i_pop]

        if self.IF_NMDA:
            for i_pop in range(self.N_POP):
                net_inputs = net_inputs + self.inputs_NMDA[i_pop]

        if self.RATE_DYN == 0:
            self.rates = TF(net_inputs, self.thresh, self.TF_NAME)
        else:
            for i_pop in range(self.N_POP):
                self.rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (
                    self.rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                    * self.EXP_DT_TAU_MEM[i_pop]
                )
                self.rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = self.rates[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] + self.DT_TAU_MEM[i_pop] * TF(
                    net_inputs[self.csumNa[i_pop] : self.csumNa[i_pop + 1]],
                    self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]],
                    self.TF_NAME,
                    self.TF_GAIN,
                )

    def update_thresh(self):
        for i_pop in range(self.N_POP):
            self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (
                self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                * self.EXP_DT_TAU_THRESH[i_pop]
            )
            # self.thresh[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = self.thresh[self.csumNa[i_pop]:self.csumNa[i_pop+1]] + self.DT_TAU_THRESH[i_pop] * self.THRESH[i_pop]
            self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (
                self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                + self.DT_TAU_THRESH[i_pop]
                * self.THRESH[i_pop]
                * self.rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
            )

    def update_ff_inputs(self):
        if ff_DYN:
            self.ff_inputs = self.ff_inputs * self.EXP_DT_TAU_FF[0]
            self.ff_inputs = self.ff_inputs + self.DT_TAU_FF[0] * self.ff_inputs_0
        else:
            self.ff_inputs = self.ff_inputs_0

    def perturb_inputs(self, step):
        if step == 0:
            self.ff_inputs_0[self.csumNa[0] : self.csumNa[0 + 1]] = 0.0

        if step == self.N_STIM_ON:
            self.ff_inputs_0[self.csumNa[0] : self.csumNa[1]] = self.Iext[0]
            # for i_pop in range(self.N_POP):
            #     self.ff_inputs_0[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = 1.1 * self.Iext[i_pop]

        if step == self.N_STIM_ON:
            if self.verbose:
                print("STIM ON")
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop])
                self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] = self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] + pertur_func(
                    theta, self.I0[i_pop], self.SIGMA0, self.PHI0, TYPE=self.PERT_TYPE
                )

        if step == self.N_STIM_OFF:
            if self.verbose:
                print("STIM OFF")
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop])
                self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] = self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] - pertur_func(
                    theta, self.I0[i_pop], self.SIGMA0, self.PHI0, TYPE=self.PERT_TYPE
                )

        if step == self.N_STIM_OFF:
            self.ff_inputs_0[self.csumNa[0] : self.csumNa[1]] = self.Iext[0]

        if step == self.N_CUE_ON:
            if self.verbose:
                print("CUE ON")
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop])
                self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] = self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] + pertur_func(
                    theta, self.I1[i_pop], self.SIGMA0, self.PHI1, TYPE=self.PERT_TYPE
                )

        if step == self.N_CUE_OFF:
            if self.verbose:
                print("CUE OFF")
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop])
                self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] = self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] - pertur_func(
                    theta, self.I1[i_pop], self.SIGMA0, self.PHI1, TYPE=self.PERT_TYPE
                )

    def generate_Cij(self):
        np.random.seed(self.SEED)

        Cij = np.zeros((self.N, self.N), dtype=np.float64)

        for i_post in range(self.N_POP):
            for j_pre in range(self.N_POP):
                Cab = generate_Cab(
                    self.Ka[j_pre],
                    self.Na[i_post],
                    self.Na[j_pre],
                    self.STRUCTURE[i_post, j_pre],
                    self.SIGMA[i_post, j_pre],
                    self.KAPPA[i_post, j_pre],
                    self.SEED,
                    self.PHASE,
                    self.verbose,
                )
                Cij[
                    self.csumNa[i_post] : self.csumNa[i_post + 1],
                    self.csumNa[j_pre] : self.csumNa[j_pre + 1],
                ] = Cab

        self.Cij = Cij

        for i_post in range(self.N_POP):
            for j_pre in range(self.N_POP):
                Cij[
                    self.csumNa[i_post] : self.csumNa[i_post + 1],
                    self.csumNa[j_pre] : self.csumNa[j_pre + 1],
                ] = (
                    Cij[
                        self.csumNa[i_post] : self.csumNa[i_post + 1],
                        self.csumNa[j_pre] : self.csumNa[j_pre + 1],
                    ]
                    * self.Jab[i_post][j_pre]
                )

        return Cij

    def generate_Cij_NMDA(self, Cij):
        Cij_NMDA = np.zeros((self.N, self.Na[0]), dtype=np.float64)

        for i_post in range(self.N_POP):
            Cij_NMDA[self.csumNa[i_post] : self.csumNa[i_post + 1]] = (
                Cij[self.csumNa[i_post] : self.csumNa[i_post + 1].copy(), : self.Na[0]]
                != 0
            ) * self.Jab_NMDA[i_post]

        return Cij_NMDA

    def update_Cij(self, Cij):
        Cab = (
            Cij[self.csumNa[0] : self.csumNa[1], self.csumNa[0] : self.csumNa[1]]
            / self.DT_TAU_SYN[0]
        )
        rE = self.rates[: self.csumNa[1]]
        Cab = numba_update_Cij(Cab, rE, ALPHA=self.ALPHA, ETA_DT=self.ETA_DT)
        Cab = Cab * self.DT_TAU_SYN[0]
        Cij[self.csumNa[0] : self.csumNa[1], self.csumNa[0] : self.csumNa[1]] = Cab
        return Cij

    def run(self):
        NE = self.Na[0]

        if self.IF_LOAD_MAT:
            Cij = np.load("Cij.npy")
        else:
            Cij = self.generate_Cij()
            if self.IF_SAVE_MAT:
                np.save("Cij.npy", Cij)

        if self.IF_NMDA:
            Cij_NMDA = np.ascontiguousarray(self.generate_Cij_NMDA(Cij))

        np.random.seed(None)

        if self.IF_STP:
            stp = STP_Model(self.Na[0], self.DT)
            if self.verbose:
                print("stp:", stp.USE, stp.TAU_REC, stp.TAU_FAC)

        if self.IF_LEARNING:
            DJij = np.zeros((self.Na[0], self.Na[0]), dtype=np.float64)
            Cij_fix = Cij[: self.Na[0], : self.Na[0]].copy()

            # theta = np.linspace(0.0, 2.0 * np.pi, self.Na[0])
            # cos_mat_learn = self.KAPPA_LEARN * np.cos(theta_mat(theta, theta))

            # N_MAPS = 10
            # Jij = numba_multiple_maps(self.Ka[0], self.Na[0], self.KAPPA[0][0], N_MAPS)
            # Cij[:self.Na[0],:self.Na[0]] = Cij[:self.Na[0],:self.Na[0]] * (1.0 + Jij)

        if self.verbose:
            self.print_params()

        running_step = 0
        data = []

        self.mean_rates = self.rates

        for step in tqdm(range(self.N_STEPS)):
            self.perturb_inputs(step)

            # self.update_ff_inputs()
            self.ff_inputs = numba_update_ff_inputs(
                self.ff_inputs,
                self.ff_inputs_0,
                self.EXP_DT_TAU_FF,
                self.DT_TAU_FF,
                self.VAR_FF,
                self.FF_DYN,
            )

            # self.update_inputs(Cij)
            self.inputs = numba_update_inputs(
                Cij,
                self.rates,
                self.inputs,
                self.csumNa,
                self.EXP_DT_TAU_SYN,
                self.SYN_DYN,
            )

            if self.IF_STP:
                self.inputs[0, : self.Na[0]] = (
                    stp.A_u_x_stp * self.inputs[0, : self.Na[0]]
                )
                stp.hansel_stp(self.rates[: self.Na[0]])
                # stp.markram_stp(self.rates[:self.Na[0]].copy())

            if self.IF_NMDA:
                self.inputs_NMDA = numba_update_inputs(
                    Cij_NMDA,
                    self.rates,
                    self.inputs_NMDA,
                    self.csumNa,
                    self.EXP_DT_TAU_NMDA,
                    self.SYN_DYN,
                )

            # self.update_rates()
            self.rates = numba_update_rates(
                self.rates,
                self.inputs,
                self.ff_inputs,
                self.inputs_NMDA,
                self.thresh,
                self.TF_NAME,
                self.TF_GAIN,
                self.csumNa,
                self.EXP_DT_TAU_MEM,
                self.DT_TAU_MEM,
                RATE_DYN=self.RATE_DYN,
                IF_NMDA=self.IF_NMDA,
                RATE_NOISE=self.RATE_NOISE,
                RATE_VAR=self.RATE_VAR,
            )

            # if step < self.N_STEADY:
            #     self.mean_rates = self.mean_rates + self.rates
            # if step == self.N_STEADY:
            #     self.mean_rates = self.mean_rates / self.N_STEADY

            if self.IF_LEARNING and step >= self.N_STEADY:
                rates_E = self.rates[: self.Na[0]].copy()
                # smooth = moving_average(rates_E, int(np.sqrt(self.K))) - np.mean(rates_E)
                smooth = moving_average(
                    rates_E - self.mean_rates[: self.Na[0]], int(self.K)
                )
                # print(smooth)

                DJij = numba_update_local_field(
                    DJij,
                    smooth,
                    self.EXP_DT_TAU_LEARN,
                    self.KAPPA_LEARN,
                    self.DT_TAU_LEARN,
                    self.ALPHA,
                )
                Cij[: self.Na[0], : self.Na[0]] = Cij_fix * (1.0 + DJij)

                # if step == self.N_STIM_ON:
                #     DJij = DJij + cos_mat_learn
                #     Cij[:self.Na[0],:self.Na[0]] = Cij_fix * (1.0 + DJij)

            if self.THRESH_DYN:
                self.update_thresh()

            running_step += 1

            if step >= self.N_STEADY:
                time = (step - self.N_STEADY) * self.ones_vec

                if running_step >= self.N_WINDOW:
                    amplitudes = []
                    phases = []

                    if self.IF_STP:
                        if self.N_POP == 2:
                            data.append(
                                np.vstack(
                                    (
                                        time,
                                        self.rates,
                                        self.ff_inputs,
                                        self.inputs,
                                        np.pad(stp.u_stp, (0, self.Na[1])),
                                        np.pad(stp.x_stp, (0, self.Na[1])),
                                        np.pad(stp.A_u_x_stp, (0, self.Na[1])),
                                    )
                                ).T
                            )
                        else:
                            data.append(
                                np.vstack(
                                    (
                                        time,
                                        self.rates,
                                        self.ff_inputs,
                                        self.inputs,
                                        stp.u_stp,
                                        stp.x_stp,
                                        stp.A_u_x_stp,
                                    )
                                ).T
                            )
                    else:
                        data.append(
                            np.vstack((time, self.rates, self.ff_inputs, self.inputs)).T
                        )

                    # try:
                    #     print('time (ms)', np.round(step/self.N_STEPS, 2),
                    #           'rates (Hz)', np.round(np.mean(self.rates[:NE]), 2),
                    #           np.round(np.mean(self.rates[NE:self.csumNa[2]]), 2),
                    #           np.round(np.mean(self.rates[self.csumNa[2]:]), 2))
                    # except:
                    if self.verbose:
                        print(
                            "time (ms)",
                            np.round(step / self.N_STEPS, 2),
                            "rates (Hz)",
                            np.round(np.mean(self.rates[:NE]), 2),
                        )

                    m1, phase = decode_bump(self.rates[: self.csumNa[1]])
                    amplitudes.append(m1)
                    phases.append(phase * 180.0 / np.pi)

                    if self.N_POP > 1:
                        m1, phase = decode_bump(
                            self.rates[self.csumNa[1] : self.csumNa[2]]
                        )
                        amplitudes.append(m1)
                        phases.append(phase * 180.0 / np.pi)

                    if self.N_POP > 2:
                        m1, phase = decode_bump(self.rates[self.csumNa[2] :])
                        amplitudes.append(m1)
                        phases.append(phase * 180.0 / np.pi)

                    if self.verbose:
                        print(
                            "m1", np.round(amplitudes, 2), "phase", np.round(phases, 2)
                        )

                    running_step = 0

        del Cij
        data = np.stack(np.array(data), axis=0)
        self.df = nd_numpy_to_nested(data, N_POP=self.N_POP, IF_STP=self.IF_STP)

        if self.SAVE:
            if self.verbose:
                print("saving data to", self.FILE_NAME + ".h5")
            store = HDFStore("../data/" + self.FILE_NAME + ".h5", "w")
            store.append("data", self.df, format="table", data_columns=True)
            store.close()

        return self


if __name__ == "__main__":
    # # set_num_threads(50)
    name = sys.argv[1]
    config = safe_load(open("../conf/config_" + name + ".yml", "r"))
    config["FILE_NAME"] = name
    model = Network(**config)

    start = perf_counter()
    model.run()
    end = perf_counter()

    print("Elapsed (with compilation) = {}s".format((end - start)))