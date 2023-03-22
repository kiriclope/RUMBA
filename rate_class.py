import time
import h5py
import yaml

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numba import jit
from configparser import ConfigParser
from tqdm import tqdm

import params

def read_df(filename):
    with pd.HDFStore(filename) as h5:
        print(h5.keys())
        df = pd.concat(map(h5.get, h5.keys()), axis=-1)
    return df

def create_df(step, rates, inputs):
    idx = np.arange(rates.shape[0])
    time = step * np.ones(rates.shape[0])

    data = np.vstack((time, idx, rates, inputs))
    df = pd.DataFrame(data.T, columns=['time','neurons', 'rates', 'h_E', 'h_I'])
    df = df.round(1)
    df = df.astype({"neurons": int})

    return df


def TF(x):
    return x * (x>0)


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


# @jit
def strided_method(ar):
    a = np.concatenate((ar, ar[:-1]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L - 1 :], (L, L), (-n, n))


# @jit(nopython=False)
def generate_Cij(K, N, STRUCTURE=None, SIGMA=1):

    Pij = 1 
    print('random connectivity')
    if STRUCTURE is not None:
        theta = np.linspace(0.0, 2.0 * np.pi, N)
        # print('theta', theta.shape)

        theta_ij = strided_method(theta)
        cos_ij = np.cos(theta_ij)
        # print('cos', cos_ij.shape)
    
    if STRUCTURE == "ring":
        print('with strong cosine structure')
        Pij = np.array(1 + SIGMA * cos_ij)

    elif STRUCTURE == "spec":
        print('with weak cosine structure')
        Pij = np.array(1 + SIGMA * cos_ij / np.sqrt(K) )
        
    Cij = np.random.rand(N, N) < K / N * Pij

    return 1.0*Cij


class Network:
    def __init__(self, **kwargs):

        const = Bunch(kwargs)

        self.filename = const.filename
        # SIMULATION
        self.DT = const.DT
        self.DURATION = float(const.DURATION)
        self.N_STEPS = int(self.DURATION / self.DT)

        self.N_STEADY = int(const.T_STEADY / self.DT)

        # PARAMETERS
        self.N = int(const.N)
        self.K = const.K

        self.Na = [int(self.N * const.frac[0]), int(self.N * const.frac[1])]
        self.Ka = [self.K * const.frac[0], self.K * const.frac[1]]

        self.TAU_SYN = const.TAU_SYN
        self.TAU_MEM = const.TAU_MEM

        self.EXP_DT_TAU_SYN = [
            np.exp(-self.DT / self.TAU_SYN[0]),
            np.exp(-self.DT / self.TAU_SYN[1]),
        ]

        self.DT_TAU_SYN = [self.DT / self.TAU_SYN[0], self.DT / self.TAU_SYN[1]]

        self.EXP_DT_TAU_MEM = [
            np.exp(-self.DT / self.TAU_MEM[0]),
            np.exp(-self.DT / self.TAU_MEM[1]),
        ]

        self.DT_TAU_MEM = [self.DT / self.TAU_MEM[0], self.DT / self.TAU_MEM[1]]

        self.M0 = const.M0

        self.Jab = np.array(const.Jab).reshape(2,2)
        self.Jab *= const.GAIN

        self.Jab[0] *= self.DT_TAU_SYN[0] / np.sqrt(self.Ka[0])
        self.Jab[1] *= self.DT_TAU_SYN[1] / np.sqrt(self.Ka[1])

        self.Jab[np.isinf(self.Jab)] = 0 

        # self.Jab[0] /= np.sqrt(self.Ka[0])
        # self.Jab[1] /= np.sqrt(self.Ka[1])

        self.Iext = const.Iext
        self.Iext[0] *= np.sqrt(self.Ka[0]) * self.M0
        self.Iext[1] *= np.sqrt(self.Ka[0]) * self.M0

        self.STRUCTURE = const.STRUCTURE
        self.SIGMA = const.SIGMA

        self.rates = np.zeros(self.N)
        self.inputs = np.zeros((2, self.N))
        self.net_inputs = np.zeros(self.N)

        self.ff_inputs = np.ones(self.N)
        self.ff_inputs[:self.Na[0]] *= self.Iext[0]
        self.ff_inputs[self.Na[0]:] *= self.Iext[1]

    def print_params(self):
        print('N', self.N, 'Na', self.Na)
        print('K', self.K, 'Ka', self.Ka)

        print('Iext', self.Iext)
        print('Jab', self.Jab)

    def update_inputs(self, Cij):
        NE = self.Na[0]

        self.inputs[0] *= self.EXP_DT_TAU_SYN[0]  # inputs from E
        self.inputs[1] *= self.EXP_DT_TAU_SYN[1]  # inputs from I

        self.inputs[0] += np.dot(Cij[:, :NE], self.rates[:NE]) 
        self.inputs[1] += np.dot(Cij[:, NE:], self.rates[NE:]) 
        
        self.net_inputs = self.ff_inputs - self.inputs[0]
        # self.net_inputs = self.ff_inputs + self.inputs[0] + self.inputs[1]

    def update_rates(self):
        self.rates = TF(self.net_inputs)

        # self.rates[:NE] *= self.EXP_DT_TAU_MEM[0]
        # self.rates[NE:] *= self.EXP_DT_TAU_MEM[1]

        # self.rates[:NE] += self.DT_TAU_MEM[0] * TF(self.net_inputs[:NE])
        # self.rates[NE:] += self.DT_TAU_MEM[1] * TF(self.net_inputs[NE:])

    def run(self):
        NE = self.Na[0]
        Cij = generate_Cij(self.K, self.N, self.STRUCTURE, self.SIGMA)
        
        Cij[:NE, :NE] *= self.Jab[0][0]
        Cij[NE:, :NE] *= self.Jab[1][0]
        
        Cij[:NE, NE:] *= self.Jab[0][1]
        Cij[NE:, NE:] *= self.Jab[1][1]
        
        store = pd.HDFStore(self.filename + '.h5', 'w')
        self.print_params()

        for step in tqdm(range(self.N_STEPS)):
            self.update_rates()
            self.update_inputs(Cij)

            # print(step, np.mean(self.rates[:NE]) * 1000, np.mean(self.rates[NE:]) * 1000)

            if step >= self.N_STEADY:
                df = create_df(step/self.N_STEPS, self.rates * 1000, self.inputs)
                store.append('data', df, format='table', data_columns=True)

        store.close()

        return self


if __name__ == "__main__":

    config = yaml.safe_load(open("./config.yml", "r"))
    # print(config.keys())
    model = Network(**config)
    
    start = time.perf_counter()
    model.run()
    end = time.perf_counter()

    print("Elapsed (with compilation) = {}s".format((end - start)))
