#!/home/leon/mambaforge/bin/python3.8
from numba import jit, int32, float32, float64, types, typed, typeof
from numba.experimental import jitclass

from configparser import ConfigParser

import yaml
import numpy as np
import matplotlib.pyplot as plt
import time

import params

class Network:
    def __init__(self, **kwargs):

        # SIMULATION
        self.DT = DT
        self.DURATION = float(DURATION)
        self.N_STEPS = int(self.DURATION / self.DT)

        # PARAMETERS
        self.N = int(N)
        self.K = K

        self.Na = [int(self.N * frac[0]), int(self.N * frac[1])]
        self.Ka = [self.K * frac[0], self.K * frac[1]]

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
        self.Jab[0] *= self.EXP_DT_TAU_SYN[0] / np.sqrt(self.Ka[0]) / self.TAU_SYN[0]
        self.Jab[1] *= self.EXP_DT_TAU_SYN[1] / np.sqrt(self.Ka[1]) / self.TAU_SYN[1]

        self.Iext = Iext
        self.Iext[0] *= np.sqrt(self.Ka[0]) * self.M0
        self.Iext[1] *= np.sqrt(self.Ka[0]) * self.M0

        self.STRUCTURE = STRUCTURE
        self.SIGMA = SIGMA
        
        self.rates = np.zeros(self.N)
        self.inputs = np.zeros((2, self.N))
        self.net_intputs = np.zeros(self.N)

        self.ff_inputs = np.ones(self.N)
        self.ff_inputs[:NE] *= self.Iext[0]
        self.ff_inputs[NE:] *= self.Iext[1]

        
    def update_inputs(self):
        self.inputs[0] *= EXP_DT_TAU_SYN[0]  # inputs from E 
        self.inputs[1] *= EXP_DT_TAU_SYN[1]  # inputs from I

        for i in range(N_NEURONS):
            # Cij, j pres to i post
            pres_E = Cij[i, :NE]
            pres_I = Cij[i, NE:]
        
            # sum over presynatptic neurons
            if i < NE:
                self.inputs[0][i] += Jab[0][0] * np.sum(self.rates[pres_E])
                self.inputs[1][i] += Jab[0][1] * np.sum(self.rates[pres_I])
            else:
                self.inputs[0][i] += Jab[1][0] * np.sum(self.rates[pres_E])
                self.inputs[1][i] += Jab[1][1] * np.sum(self.rates[pres_I])
        
        self.net_inputs = self.ff_inputs + np.sum(self.inputs, axis=0) 
        
    def update_rates(self):
        NE = self.NE
        
        self.rates[:NE] *= self.EXP_DT_TAU_MEM[0]
        self.rates[NE:] *= self.EXP_DT_TAU_MEM[1]
        
        self.rates[:NE] += self.DT_TAU_MEM[0] * TF(self.net_inputs[:NE])
        self.rates[NE:] += self.DT_TAU_MEM[1] * TF(self.net_inputs[NE:])

    def run(self):        
        # generate_Cij(self.N, self.K, self.STRUCTURE, self.SIGMA)
        
        for step in range(self.N_STEPS):
            self.update_rates()
            self.update_inputs()
        

@jit
def strided_method(ar):
    a = np.concatenate((ar, ar[:-1]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L - 1 :], (L, L), (-n, n))


@jit(nopython=True, parallel=True)
def generate_Cij(K, N, structure=None, sigma=0.25):

    if structure is None:
        Cij = np.random.rand(N, N) < K / N
    elif structure == "ring":
        theta = np.arange(0.0, 2.0 * np.pi, 1.0 / np.float32(N))
        theta_ij = strided_method(theta)
        cos_ij = np.cos(theta_ij)
        
        Cij = np.random.rand(N, N) < K / N * (1 + np.array(sigma) * cos_ij)

    return Cij


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
    
    Jab[0] *= EXP_DT_TAU_SYN[0] / np.sqrt(Ka[0]) / TAU_SYN[0]
    Jab[1] *= EXP_DT_TAU_SYN[1] / np.sqrt(Ka[1]) / TAU_SYN[1]
    
    Iext[0] *= np.sqrt(Ka[0]) * M0
    Iext[1] *= np.sqrt(Ka[0]) * M0


def initialize():
    volts = np.zeros(N)
    inputs = np.zeros((2, N))
    spiked = np.zeros(N)

    NE = int(Na[0])
    ff_inputs = np.ones(N)
    ff_inputs[:NE] *= Iext[0]
    ff_inputs[NE:] *= Iext[1]


def update_inputs(inputs, Cij):
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
    
def save_results():
    return

def run():
    
    generate_Cij()
    
    initialize()
    
    # for step in range(N_STEPS):
    #     update_rates()
    #     update_inputs()
    

if __name__ == "__main__":

    config = yaml.safe_load(open("./config.yml", "r"))
    print(config)
    run()
    
    # model = LifNetwork(**config)
    
    # start = time.perf_counter()
    # model.run()
    # end = time.perf_counter()

    # print("Elapsed (with compilation) = {}s".format((end - start)))

    # start = time.perf_counter()
    # dict = {"sigma": 0.25}

    # Mat = connectivity(200, 1000, structure="ring", sigma=0.25)
    # end = time.perf_counter()

    # print("Elapsed (with compilation) = {}s".format((end - start)))

    # plt.imshow(Mat)
