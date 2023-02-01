#!/home/leon/mambaforge/bin/python3.8
from numba import jit, int32, float32, types, typed, typeof
from numba.experimental import jitclass

from configparser import ConfigParser

import yaml
import numpy as np
import matplotlib.pyplot as plt
import time

import params


@jit
def strided_method(ar):
    a = np.concatenate((ar, ar[:-1]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L - 1 :], (L, L), (-n, n))


@jit(nopython=True, parallel=True)
def connectivity(K, N, structure=None, sigma=0.25, IF_RET=0):

    if structure is None:
        Cij = np.random.rand(N, N) < K / N
    elif structure == "ring":
        theta = np.linspace(0, 2 * np.pi, N)
        theta_ij = strided_method(theta)
        cos_ij = np.cos(theta_ij)

        Cij = np.random.rand(N, N) < K / N * (1 + sigma * cos_ij)

    # id_post = []
    # n_post = np.zeros(N)

    # for j in range(N):  # pre
    #     for i in range(N):  # post
    #         if Cij[i, j]:
    #             id_post.append(i)  # id of post for pre j
    #             n_post[j] += 1

    # id_post = np.array(id_post)
    # idx_post = np.zeros(N)
    # for i in range(N - 1):
    #     idx_post[i + 1] = idx_post[i] + n_post[i]

    return Cij
    # return Cij, id_post, idx_post, n_post


@jit(nopython=True, parallel=True)
def input_update(inputs, spiked, Cij):

    inputs[0] *= EXP_DT_TAU[0]  # inputs from pres E onto all
    inputs[1] *= EXP_DT_TAU[1]  # inputs from pres I onto all

    for j in np.where(spiked):

        # Cij[:,j] -> list of postsyn of pres neuron j
        post_E = Cij[:NE, j]
        post_I = Cij[NE:, j]

        if j < NE:
            inputs[0, post_E] += J[0, 0]
            inputs[0, post_I] += J[1, 0]
        else:
            inputs[1, post_E] += J[0, 1]
            inputs[1, post_I] += J[1, 1]

    return inputs


@jit
def volt_update(volt):

    volt[:NE] *= EXP_DT_TAU_MEM[0]
    volt[NE:] *= EXP_DT_TAU_MEM[1]

    volt[:NE] += DT_TAU[0] * (net_inputs[:NE] + VL[0])
    volt[NE:] += DT_TAU[1] * (net_inputs[NE:] + VL[1])

    return volt


@jit
def run():

    Cij = connectivity(K, N)

    volt, inputs, spiked = init()
    inputs = input_update(inputs, spiked, Cij)
    net_inputs = ff_inputs + inputs[0] + inputs[1]

    for i in range(DURATION):

        volt = volt_update(volt, net_inputs, **params)

        spiked = volt > V_TH

        inputs = input_update(inputs, spiked, Cij)

        net_inputs = ff_inputs + inputs[0] + inputs[1]


@jit
def init():
    volt = np.zeros(N)
    inputs = np.zeros((2, NE))
    spiked = np.zeros(N)

    return volt, inputs, spiked


spec = [
    # ("config", types.DictType(*(types.unicode_type, types.float32[:]))),
    ("DT", float32),
    ("DURATION", float32),
    ("N_STEPS", float32),
    ("N", int32),
    ("K", float32),
    ("Na", int32[:]),
    ("Ka", int32[:]),
    ("VL", typeof([1.0, 1.0])),
    ("V_TH", typeof([1.0, 1.0])),
    ("TAU_SYN", typeof([1.0, 1.0])),
    ("TAU_MEM", typeof([1.0, 1.0])),
    ("EXP_DT_TAU_MEM", typeof([1.0, 1.0])),
    ("DT_TAU_MEM", typeof([1.0, 1.0])),
    ("EXP_DT_TAU_SYN", typeof([1.0, 1.0])),
    ("TAU_MEM", typeof([1.0, 1.0])),
    ("Jab", typeof([1.0, 1.0])),
    ("Iext", typeof([1.0, 1.0])),
    ("M0", float32),
    ("STRUCTURE", types.unicode_type),
    ("SIGMA", typeof([1.0, 1.0])),
    ("volts", typeof([1.0, 1.0])),
    ("spiked", typeof([1.0, 1.0])),
    ("inputs", float32[:, :]),
    ("net_inputs", typeof([1.0, 1.0])),
    ("ff_inputs", typeof(np.ones(1))),
]


@jitclass(spec)
class LifNetwork:
    def __init__(
        self,
        DT,
        DURATION,
        N,
        K,
        frac,
        VL,
        V_TH,
        TAU_SYN,
        TAU_MEM,
        Jab,
        Iext,
        M0,
        STRUCTURE,
        SIGMA,
    ):

        print(DURATION)
        # self.config = config
        # print(self.config)

        self.DT = DT
        self.DURATION = float(DURATION)
        self.N_STEPS = int(self.DURATION / self.DT)

        self.N = int(N)
        self.K = K

        # self.Na =
        # print(self.Na)

        # self.Ka = [self.K * frac[0], self.K * frac[1]]
        # print(self.Ka)

        self.VL = VL
        self.V_TH = V_TH

        self.TAU_SYN = TAU_SYN
        self.TAU_MEM = TAU_MEM

        self.EXP_DT_TAU_SYN = [
            np.exp(-self.DT / self.TAU_SYN[0]),
            np.exp(-self.DT / self.TAU_SYN[1]),
        ]
        self.EXP_DT_TAU_MEM = [
            np.exp(-self.DT / self.TAU_MEM[0]),
            np.exp(-self.DT / self.TAU_MEM[1]),
        ]
        self.DT_TAU_MEM = [self.DT / self.TAU_MEM[0], self.DT / self.TAU_MEM[1]]

        self.M0 = M0

        self.Jab = Jab
        self.Jab[0] *= self.EXP_DT_TAU_SYN[0] / np.sqrt(self.K) / self.TAU_SYN[0]
        self.Jab[1] *= self.EXP_DT_TAU_SYN[1] / np.sqrt(self.K) / self.TAU_SYN[1]

        self.Iext = Iext
        # self.Iext[0] = self.Iext[0] * np.sqrt(self.Ka[0]) * self.M0
        # self.Iext[1] = self.Iext[1] * np.sqrt(self.Ka[1]) * self.M0

        self.STRUCTURE = STRUCTURE
        self.SIGMA = SIGMA

    def generate_Cij(self):
        N = self.N
        K = self.K

        if self.STRUCTURE == "None":
            self.Cij = np.random.rand(N, N) < K / N

        elif self.STRUCTURE == "RING":
            self.theta = np.linspace(0, 2 * np.pi, N)

            theta_ij = strided_method(theta)
            cos_ij = np.cos(theta_ij)

            self.Cij = np.random.rand(N, N) < K / N * (1 + self.SIGMA * cos_ij)

        return self

    def initialize(self):
        # NE = int(self.Na[0])
        self.volts = np.zeros(self.N)
        self.inputs = np.zeros((2, self.N))
        self.spiked = np.zeros(self.N)

        # self.ff_inputs = np.ones(self.N)
        # self.ff_inputs[:NE] *= self.I0[0]
        # self.ff_inputs[NE:] *= self.I0[1]

        return self

    def update_inputs(self):
        NE = int(self.Na[0])
        Jab = self.Jab

        self.inputs[0] *= self.EXP_DT_TAU_SYN[0]  # inputs from pres E onto all
        self.inputs[1] *= self.EXP_DT_TAU_SYN[1]  # inputs from pres I onto all

        for j in np.where(self.spiked):

            # Cij[:,j] -> list of postsyn of pres neuron j
            post_E = self.Cij[:NE, j]
            post_I = self.Cij[NE:, j]

            if j < NE:
                self.inputs[0, post_E] += Jab[0, 0]
                self.inputs[0, post_I] += Jab[1, 0]
            else:
                self.inputs[1, post_E] += Jab[0, 1]
                self.inputs[1, post_I] += Jab[1, 1]

        self.net_inputs = self.ff_inputs + self.inputs[0] + self.inputs[1]

        return self

    def update_volts(self):
        NE = int(self.Na[0])
        VL = self.VL

        self.volts[:NE] *= self.EXP_DT_TAU_MEM[0]
        self.volts[NE:] *= self.EXP_DT_TAU_MEM[1]

        self.volts[:NE] += self.DT_TAU_MEM[0] * (self.net_inputs[:NE] + VL[0])
        self.volts[NE:] += self.DT_TAU_MEM[1] * (self.net_inputs[NE:] + VL[1])

        return self

    def save_results(self):
        return self

    def run(self):
        # self.read_config()

        # self.generate_Cij()
        self.initialize()
        # self.update_inputs()

        # for step in range(self.N_STEPS):
        #     self.update_volts()
        #     self.spiked = self.volts > self.V_TH
        #     self.update_inputs()

        return self


if __name__ == "__main__":

    config = yaml.load(open("./config.yml", "r"))
    model = LifNetwork(**config)

    start = time.perf_counter()
    model.run()
    end = time.perf_counter()

    print("Elapsed (with compilation) = {}s".format((end - start)))

    # start = time.perf_counter()
    # dict = {"sigma": 0.25}

    # Mat = connectivity(200, 1000, structure="ring", sigma=0.25)
    # end = time.perf_counter()

    # print("Elapsed (with compilation) = {}s".format((end - start)))

    # plt.imshow(Mat)
