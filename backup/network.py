#!/home/leon/mambaforge/bin/python3.8
import time
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import params
import yaml
from numba import float32, float64, int32, jit, typed, typeof, types
from numba.experimental import jitclass


@jit
def strided_method(ar):
    a = np.concatenate((ar, ar[:-1]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L - 1 :], (L, L), (-n, n))


@jit(nopython=True, parallel=True)
def generate_Cij(K, N, structure=None, sigma=0.25):

    # if structure is None:
    Cij = np.random.rand(N, N) < K / N
    # elif structure == "ring":
    #     theta = np.arange(0.0, 2.0 * np.pi, 1.0 / np.float32(N))
    #     theta_ij = strided_method(theta)
    #     cos_ij = np.cos(theta_ij)

    #     Cij = np.random.rand(N, N) < K / N * (1 + np.array(sigma) * cos_ij)

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


def init_params():

        N_STEPS = int(DURATION / DT)
        Na = [int(N * frac[0]), int(N * frac[1])]
        Ka = [K * frac[0], K * frac[1]]

        EXP_DT_TAU_SYN = [
            np.exp(-DT / TAU_SYN[0]),
            np.exp(-DT / TAU_SYN[1]),
        ]

        EXP_DT_TAU_MEM = [
            np.exp(-DT / TAU_MEM[0]),
            np.exp(-DT / TAU_MEM[1]),
        ]

        DT_TAU_MEM = [DT / TAU_MEM[0], DT / TAU_MEM[1]]

        Jab[0] *= EXP_DT_TAU_SYN[0] / np.sqrt(K) / TAU_SYN[0]
        Jab[1] *= EXP_DT_TAU_SYN[1] / np.sqrt(K) / TAU_SYN[1]

        Iext[0] *= np.sqrt(Ka[0]) * M0
        Iext[1] *= np.sqrt(Ka[1]) * M0

def initialize():
    volts = np.zeros(N)
    inputs = np.zeros((2, N))
    spiked = np.zeros(N)

    NE = int(Na[0])
    ff_inputs = np.ones(N)
    ff_inputs[:NE] *= Iext[0]
    ff_inputs[NE:] *= Iext[1]

    return

def update_inputs():
    inputs[0] *= EXP_DT_TAU_SYN[0]  # inputs from E 
    inputs[1] *= EXP_DT_TAU_SYN[1]  # inputs from I

    for i in range(N_NEURONS):
        # Cij, j pres to i post
        pres_E = Cij[i, :NE]
        pres_I = Cij[i, NE:]
        
        # sum over presynatptic neurons
        if i < NE:
            inputs[0][i] += J[0][0] * np.sum(rates[pres_E])
            inputs[1][i] += J[0][1] * np.sum(rates[pres_I])
        else:
            inputs[0][i] += J[1][0] * np.sum(rates[pres_E])
            inputs[1][i] += J[1][1] * np.sum(rates[pres_I])
    
    net_inputs = ff_inputs + inputs[0] + inputs[1]
    
    return net_inputs 

def update_rates():

    rates[:NE] *= EXP_DT_TAU_MEM[0]
    rates[NE:] *= EXP_DT_TAU_MEM[1]
    
    rates[:NE] += DT_TAU_MEM[0] * TF(net_inputs[:NE])
    rates[NE:] += DT_TAU_MEM[1] * TF(net_inputs[NE:])
   
    return rates
    
def update_post_inputs():
    NE = int(Na[0])
    Jab = Jab

    inputs[0] *= EXP_DT_TAU_SYN[0]  # inputs from E 
    inputs[1] *= EXP_DT_TAU_SYN[1]  # inputs from I

    for j in np.where(spiked):

    # Cij[:,j] -> list of postsyn of pres neuron j
    post_E = Cij[:NE, j]
    post_I = Cij[NE:, j]

    if j < NE:
        inputs[0, post_E] += Jab[0, 0]
        inputs[0, post_I] += Jab[1, 0]
    else:
        inputs[1, post_E] += Jab[0, 1]
        inputs[1, post_I] += Jab[1, 1]

    net_inputs = ff_inputs + inputs[0] + inputs[1]

    return

def update_volts():
    NE = int(Na[0])
    VL = VL

    volts[:NE] *= EXP_DT_TAU_MEM[0]
    volts[NE:] *= EXP_DT_TAU_MEM[1]

    volts[:NE] += DT_TAU_MEM[0] * (net_inputs[:NE] + VL[0])
    volts[NE:] += DT_TAU_MEM[1] * (net_inputs[NE:] + VL[1])

    return

def save_results():
    return

def run():
    generate_Cij(N, K, STRUCTURE, SIGMA)
    initialize()
    # update_inputs()

    # for step in range(N_STEPS):
    #     update_volts()
    #     spiked = volts > V_TH
    #     update_inputs()

    return


if __name__ == "__main__":

    config = yaml.safe_load(open("./config.yml", "r"))
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
