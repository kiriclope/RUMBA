import numpy as np

from time import perf_counter
from yaml import safe_load
from configparser import ConfigParser
from tqdm import tqdm
from scipy.special import erf
from pandas import DataFrame, HDFStore

def create_df(time, idx, rates, ff_inputs, inputs):

    data = np.vstack((time, idx, rates, ff_inputs, inputs))
    df = DataFrame(data.T,
        columns=['time','neurons', 'rates', 'ff', 'h_E', 'h_I'])
    # df = df.round(2)
    df = df.astype({"neurons": int})

    return df


def TF(x, tfname='TL'):
    if tfname=='TL':
        return x * (x > 0.0)
    elif tfname=='Sig':
        return 0.5 * (1.0 + erf( x / np.sqrt(2.0)))
    elif tfname=='LIF':
        return - 1.0 * (x > 1.0) / np.log(1.0 - 1.0 / x)


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def theta_mat(theta):
    theta_mat = np.zeros((theta.shape[0], theta.shape[0]))
    for i in range(theta.shape[0]):
        for j in range(theta.shape[0]):
            theta_mat[i,j] = theta[i] - theta[j]

    return theta_mat

# @jit
def strided_method(ar):
    a = np.concatenate((ar, ar[1:]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L-1:], (L, L), (-n, n))


# @jit(nopython=False)
def generate_Cij(K, N, STRUCTURE=None, SIGMA=1, SEED=None):

    Pij = 1.0

    print('random connectivity')
    if STRUCTURE != 'None':
        theta = np.linspace(0.0, 2.0 * np.pi, N)
        # print('theta', theta.shape)
        theta_ij = strided_method(theta).T
        # theta_ij = theta_mat(theta)
        cos_ij = np.cos(theta_ij)
        # print('cos', cos_ij.shape)

    if STRUCTURE == "ring":
        print('with strong cosine structure')
        Pij = 1.0 + SIGMA * cos_ij

    elif STRUCTURE == "spec":
        print('with weak cosine structure')
        Pij = 1.0 + SIGMA * cos_ij / np.sqrt(K)

    elif STRUCTURE == "low_rank":
        print('with weak low rank structure')
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]
        rng = np.random.default_rng(seed=None)
        ksi = rng.multivariate_normal(mean, cov, size=N).T
        Pij = 1.0 + SIGMA * ksi[0] * ksi[1] / np.sqrt(K)

    Cij = np.random.rand(N, N) < (K / N) * Pij

    return Cij


class Network:
    def __init__(self, **kwargs):

        const = Bunch(kwargs)

        self.verbose = const.verbose
        if self.verbose:
            print(kwargs.keys())

        self.FILE_NAME = const.FILE_NAME
        self.TF_NAME = const.TF_NAME
        
        # SIMULATION
        self.DT = const.DT
        self.DURATION = float(const.DURATION)
        self.N_STEPS = int(self.DURATION / self.DT)

        self.N_STEADY = int(const.T_STEADY / self.DT)
        self.N_WINDOW = int(const.T_WINDOW / self.DT)

        # PARAMETERS
        self.N = int(const.N)
        self.K = const.K

        self.idx = np.arange(self.N)
        self.idx.astype(np.float32)
        self.ones_vec = np.ones((self.N,), dtype=np.float32) / self.N_STEPS

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

        self.Jab = np.array(const.Jab, dtype=np.float32).reshape(2,2)
        self.Jab *= const.GAIN

        self.Jab[:, 0] = self.Jab[:, 0] * self.DT_TAU_SYN[0] / np.sqrt(self.Ka[0])
        self.Jab[:, 1] = self.Jab[:, 1] * self.DT_TAU_SYN[1] / np.sqrt(self.Ka[1])
        
        self.Jab[np.isinf(self.Jab)] = 0        
        
        self.Iext = const.Iext
        self.Iext[0] *= np.sqrt(self.Ka[0]) * self.M0
        self.Iext[1] *= np.sqrt(self.Ka[0]) * self.M0
        
        self.SEED = const.SEED
        if self.SEED == 'None':
            self.SEED = None

        self.STRUCTURE = const.STRUCTURE
        self.SIGMA = const.SIGMA
                
        self.rates = np.zeros( (self.N,), dtype=np.float32)
        self.inputs = np.zeros((2, self.N), dtype=np.float32)

        rng = np.random.default_rng()
        self.rates[:self.Na[0]] = rng.normal(10, 1, self.Na[0])
        self.rates[self.Na[0]:] = rng.normal(10, 1, self.Na[1])
        
        self.ff_inputs = np.ones((self.N,), dtype=np.float32)
        self.ff_inputs[:self.Na[0]] = self.ff_inputs[:self.Na[0]] * self.Iext[0]
        self.ff_inputs[self.Na[0]:] = self.ff_inputs[self.Na[0]:] * self.Iext[1]

    def print_params(self):
        print('Parameters:')
        print('N', self.N, 'Na', self.Na)
        print('K', self.K, 'Ka', self.Ka)

        print('Iext', self.Iext)
        print('Jab', self.Jab.flatten())

        print('saving data to', self.FILE_NAME + '.h5')

    def update_inputs(self, Cij):
        NE = self.Na[0]

        self.inputs[0] = self.inputs[0] * self.EXP_DT_TAU_SYN[0]  # inputs from E
        self.inputs[0] = self.inputs[0] + np.dot(Cij[:, :NE], self.rates[:NE])
        
        if self.Na[1]>0:
            self.inputs[1] = self.inputs[1] * self.EXP_DT_TAU_SYN[1]  # inputs from I
            self.inputs[1] = self.inputs[1] + np.dot(Cij[:, NE:], self.rates[NE:])
            
    def update_rates(self):
        NE = self.Na[0]
        net_inputs = self.ff_inputs
        
        net_inputs = net_inputs + self.inputs[0]        
        if self.Na[1]>0:
            net_inputs = net_inputs + self.inputs[1]        
        
        self.rates = TF(net_inputs, self.TF_NAME)

        # self.rates[:NE] *= self.EXP_DT_TAU_MEM[0] # excitatory rates
        # self.rates[:NE] += self.DT_TAU_MEM[0] * TF(net_inputs[:NE], self.TF_NAME)
        
        # if self.Na[1]>0:
        #     self.rates[NE:] *= self.EXP_DT_TAU_MEM[1]  # inhibitory rates
        #     self.rates[NE:] += self.DT_TAU_MEM[1] * TF(net_inputs[NE:], self.TF_NAME)
        
    def run(self):
        NE = self.Na[0]
        Cij = generate_Cij(self.K, self.N,
            self.STRUCTURE, self.SIGMA, self.SEED)

        Cij = 1.0 * Cij
        Cij[:NE, :NE] = Cij[:NE, :NE] * self.Jab[0][0]
        
        if self.Na[1]>0:        
            Cij[NE:, :NE] = Cij[NE:, :NE] * self.Jab[1][0]
            Cij[:NE, NE:] = Cij[:NE, NE:] * self.Jab[0][1]
            Cij[NE:, NE:] = Cij[NE:, NE:] * self.Jab[1][1]

        Cij.astype(np.float32)
        store = HDFStore(self.FILE_NAME + '.h5', 'w')
        self.print_params()
        
        running_step = 0
        for step in tqdm(range(self.N_STEPS)):
            self.update_rates()
            self.update_inputs(Cij)

            running_step += 1

            if step >= self.N_STEADY:
                time = step * self.ones_vec
                df = create_df(time, self.idx, self.rates * 1000, self.ff_inputs, self.inputs)
                store.append('data', df, format='table', data_columns=True)

                if running_step >= self.N_WINDOW:
                    print('time (ms)', np.round(step/self.N_STEPS, 2),
                          'rates (Hz)', np.round(np.mean(self.rates[:NE]) * 1000, 2),
                          np.round(np.mean(self.rates[NE:]) * 1000, 2))
                    running_step = 0

        store.close()

        return self


if __name__ == "__main__":

    config = safe_load(open("./config.yml", "r"))
    model = Network(**config)

    start = perf_counter()
    model.run()
    end = perf_counter()

    print("Elapsed (with compilation) = {}s".format((end - start)))
