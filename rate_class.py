import numpy as np
from time import perf_counter
from yaml import safe_load
from configparser import ConfigParser
from tqdm import tqdm
from scipy.special import erf
from pandas import DataFrame, HDFStore, concat
from numba import jit


def nd_numpy_to_nested(X):
    """Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints)
    into pandas DataFrame (with time series as pandas Series in cells)
    Parameters
    ----------
    X : NumPy ndarray, input
    
    Returns
    -------
    pandas DataFrame
    """
    print(X.shape)
    
    variables = ['time','neurons', 'rates', 'ff', 'h_E', 'h_I']
    df = DataFrame()
    idx = np.arange(0, X.shape[1], 1)
    for i_time in range(X.shape[0]):            
        df_i = DataFrame( X[i_time,:, 1:], columns = ['rates', 'ff', 'h_E', 'h_I'])
        df_i['neurons'] = idx
        df_i['time'] = X[i_time, 0, 0]

        # print(df_i)
        df = concat((df, df_i))
        
    print(df)
    
    return df


def create_df(data):
    print(data.shape)
    df = DataFrame(data.T,
        columns=['time','neurons', 'rates', 'ff', 'h_E', 'h_I'])

    df = DataFrame(['time','neurons', 'rates', 'ff', 'h_E', 'h_I'])
    # df = df.round(2)
    df = df.astype({"neurons": int})

    return df


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def TF(x, tfname='TL'):
    if tfname=='TL':
        x[x>=50]==50
        return x * (x > 0.0)
    # elif tfname=='Sig':
    #     return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    # elif tfname=='LIF':
    #     return - 1.0 * (x > 1.0) / np.log(1.0 - 1.0 / x)


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def theta_mat(theta, phi):
    theta_mat = np.zeros((phi.shape[0], theta.shape[0]))
    
    for i in range(phi.shape[0]):
        for j in range(theta.shape[0]):
            theta_mat[i, j] = phi[i] - theta[j]

    return theta_mat


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def strided_method(ar):
    a = np.concatenate((ar, ar[1:]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L-1:], (L, L), (-n, n))


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def generate_Cab(Kb, Na, Nb, STRUCTURE=None, SIGMA=1, SEED=None, PHASE=0):

    Pij = np.zeros((Na, Nb), dtype=np.float32)
    Cij = np.zeros((Na, Nb), dtype=np.int32)
    
    print('random connectivity')
    if STRUCTURE != 'None':
        theta = np.linspace(0.0, 2.0 * np.pi, Nb)
        theta = theta.astype(np.float32)

        phi = np.linspace(0.0, 2.0 * np.pi, Na)
        phi = phi.astype(np.float32)
        
        theta_ij = theta_mat(theta, phi)
        cos_ij = np.cos(theta_ij + PHASE)
        
        if 'lateral' in STRUCTURE:
            cos2_ij = np.cos(2.0 * theta_ij)
            print('lateral')
            Pij[:, :] = cos_ij + cos2_ij 
        else:
            Pij[:, :] = cos_ij 
        
    if "ring" in STRUCTURE:
        print('with strong cosine structure')
        Pij[:, :] = Pij[:, :] * np.float32(SIGMA)
        
    elif "spec" in STRUCTURE:
        print('with weak cosine structure')
        Pij[:, :] = Pij[:, :] * np.float32(SIGMA) / np.sqrt(Kb)

    elif "small" in STRUCTURE:
        print('with very weak cosine structure')
        Pij[:, :] = Pij[:, :] * np.float32(SIGMA) / Kb
        
    elif "dense" in STRUCTURE:
        print('with dense cosine structure')
        Pij[:, :] = Pij[:, :] * np.float32(SIGMA) / Kb * np.sqrt(Nb) 
            
    Pij[:, :] = Pij[:, :] + 1.0 
    Cij = (np.random.rand(Na, Nb) < (Kb / Nb) * Pij)
    
    return Cij


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def generate_Cij(Ka, Na, STRUCTURE=None, SIGMA=1, SEED=None):

    N = Na[0]+Na[1]
    K = Ka[0]+Ka[1]
    Pij = np.zeros((N,N), dtype=np.float32)
    Cij = np.zeros((N,N), dtype=np.int32)

    # print('random connectivity')
    if STRUCTURE != 'None':
        theta = np.linspace(0.0, 2.0 * np.pi, Na[0])
        theta = theta.astype(np.float32)
        # print('theta', theta.shape)
        theta_ij = strided_method(theta).T
        # theta_ij = theta_mat(theta)
        cos_ij = np.cos(theta_ij)
        cos2_ij = np.cos(2.0 * theta_ij)
        
        # print('cos', cos_ij.shape)
        Pij[:Na[0], :Na[0]] = cos_ij + cos2_ij

    if STRUCTURE == "ring":
        print('with strong cosine structure')
        Pij[:Na[0], :Na[0]] = Pij[:Na[0], :Na[0]] * SIGMA
        
    elif STRUCTURE == "spec":
        print('with weak cosine structure')
        Pij[:Na[0], :Na[0]] = Pij[:Na[0], :Na[0]] * SIGMA / np.sqrt(K)

    Pij = Pij + 1
    # elif STRUCTURE == "low_rank":
    #     print('with weak low rank structure')
    #     mean = [0, 0]
    #     cov = [[1, 0], [0, 1]]
    #     # rng = np.random.default_rng(seed=None)
    #     # ksi = rng.multivariate_normal(mean, cov, size=N).T
    #     # ksi = np.random.multivariate_normal(mean, cov, size=N).T
    #     # Pij = 1.0 + SIGMA * ksi[0] * ksi[1] / np.sqrt(K)

    Cij = 1.0 * (np.random.rand(N, N) < (K / N) * Pij)

    return Cij


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_inputs(Cij, rates, inputs, Na, EXP_DT_TAU_SYN):
    NE = Na[0]

    inputs[0] = inputs[0] * EXP_DT_TAU_SYN[0]  # inputs from E
    CaE = Cij[:, :NE]
    rE = rates[:NE]
    inputs[0] = inputs[0] + np.dot(CaE, rE)
        
    if Na[1]>0:
        CaI = Cij[:, NE:]
        rI = rates[NE:]
        inputs[1] = inputs[1] * EXP_DT_TAU_SYN[1]  # inputs from I
        inputs[1] = inputs[1] + np.dot(CaI, rI)

    return inputs


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_rates(rates, inputs, ff_inputs, Na, TF_NAME):
    NE = Na[0]
    net_inputs = ff_inputs
    
    net_inputs = net_inputs + inputs[0]        
    if Na[1]>0:
        net_inputs = net_inputs + inputs[1]     
        
    rates = TF(net_inputs, TF_NAME)

    return rates


class Network:
    def __init__(self, **kwargs):

        const = Bunch(kwargs)

        self.SAVE = const.SAVE
        self.verbose = const.verbose
        if self.verbose:
            print(kwargs.keys())

        self.FILE_NAME = const.FILE_NAME
        self.TF_NAME = const.TF_NAME

        self.RATE_DYN = const.RATE_DYN
        # SIMULATION
        self.DT = const.DT
        self.DURATION = float(const.DURATION)
        self.N_STEPS = int(self.DURATION / self.DT)

        self.N_STEADY = int(const.T_STEADY / self.DT)
        self.N_WINDOW = int(const.T_WINDOW / self.DT)

        self.N_STIM_ON = int(const.T_STIM_ON / self.DT)
        self.N_STIM_OFF = int(const.T_STIM_OFF / self.DT)

        # PARAMETERS
        self.N = int(const.N)
        self.K = const.K

        self.idx = np.arange(self.N)
        self.idx.astype(np.float32)
        self.ones_vec = np.ones((self.N,), dtype=np.float32) / self.N_STEPS

        self.Na = np.array([int(self.N * const.frac[0]), int(self.N * const.frac[1])], dtype=np.int64)
        self.Ka = np.array([self.K * const.frac[0], self.K * const.frac[1]], dtype=np.float32)
        
        self.TAU_SYN = const.TAU_SYN 
        self.TAU_MEM = const.TAU_MEM

        self.EXP_DT_TAU_SYN = np.array( [
            np.exp(-self.DT / self.TAU_SYN[0]),
            np.exp(-self.DT / self.TAU_SYN[1]),
        ], dtype=np.float32)

        self.DT_TAU_SYN = np.array([self.DT / self.TAU_SYN[0], self.DT / self.TAU_SYN[1]], dtype=np.float32)

        self.EXP_DT_TAU_MEM = np.array([
            np.exp(-self.DT / self.TAU_MEM[0]),
            np.exp(-self.DT / self.TAU_MEM[1]),
        ], dtype=np.float32)

        self.DT_TAU_MEM = np.array([self.DT / self.TAU_MEM[0], self.DT / self.TAU_MEM[1]], dtype=np.float32)

        self.M0 = const.M0

        self.Jab = np.array(const.Jab, dtype=np.float32).reshape(2,2)
        self.GAIN = const.GAIN
        self.Jab *= const.GAIN
        self.Iext = const.Iext

        self.mf_rates = -np.dot(np.linalg.inv(self.Jab), self.Iext)
 
        self.Jab[:, 0] = self.Jab[:, 0] * self.DT_TAU_SYN[0] / np.sqrt(self.Ka[0])
        self.Jab[:, 1] = self.Jab[:, 1] * self.DT_TAU_SYN[1] / np.sqrt(self.Ka[1])
        
        self.Jab[np.isinf(self.Jab)] = 0        
        
        self.Iext[0] *= np.sqrt(self.Ka[0]) * self.M0
        self.Iext[1] *= np.sqrt(self.Ka[0]) * self.M0

        self.I0 = const.I0 * self.M0 * np.sqrt(self.Ka[0])
        self.PHI0 = np.random.uniform(2*np.pi)
        
        self.SEED = const.SEED
        if self.SEED == 'None':
            self.SEED = None

        self.STRUCTURE = const.STRUCTURE
        self.SIGMA = const.SIGMA
                
        self.rates = np.zeros( (self.N,), dtype=np.float32)
        self.inputs = np.zeros((2, self.N), dtype=np.float32)

        # rng = np.random.default_rng()
        # self.rates[:self.Na[0]] = rng.normal(5, 1, self.Na[0])
        # self.rates[self.Na[0]:] = rng.normal(10, 2, self.Na[1])
        
        self.ff_inputs = np.ones((self.N,), dtype=np.float32)
        self.ff_inputs[:self.Na[0]] = self.ff_inputs[:self.Na[0]] * self.Iext[0]
        self.ff_inputs[self.Na[0]:] = self.ff_inputs[self.Na[0]:] * self.Iext[1]

    def print_params(self):
        print('Parameters:')
        print('N', self.N, 'Na', self.Na)
        print('K', self.K, 'Ka', self.Ka)

        print('Iext', self.Iext)
        print('Jab', self.Jab.flatten())

        print('MF Rates:', self.mf_rates)

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

        if self.RATE_DYN==0:
            self.rates = TF(net_inputs, self.TF_NAME)
        else:
            self.rates[:NE] = self.rates[:NE] * self.EXP_DT_TAU_MEM[0] # excitatory rates
            self.rates[:NE] = self.rates[:NE] + self.DT_TAU_MEM[0] * TF(net_inputs[:NE], self.TF_NAME)
        
            if self.Na[1]>0:
                self.rates[NE:] = self.rates[NE:] * self.EXP_DT_TAU_MEM[1]  # inhibitory rates
                self.rates[NE:] = self.rates[NE:] + self.DT_TAU_MEM[1] * TF(net_inputs[NE:], self.TF_NAME)

    def perturb_inputs(self, step):
        NE = self.Na[0]
        if step == self.N_STIM_ON:
            print('CUE ON')
            theta = np.linspace(0.0, 2.0 * np.pi, NE) 
            self.ff_inputs[:NE] = self.ff_inputs[:NE] + self.I0 * (1.0 + np.cos(theta - self.PHI0) )
    
        if step == self.N_STIM_OFF:
            print('CUE OFF')
            theta = np.linspace(0.0, 2.0 * np.pi, NE)
            self.ff_inputs[:NE] = self.ff_inputs[:NE] - self.I0 * (1.0 + np.cos(theta - self.PHI0) )
            
    def generate_Cij(self):
        NE = self.Na[0]        
        Cij = np.zeros((self.N, self.N), dtype=np.float32)

        Cee = generate_Cab(self.Ka[0], self.Na[0], self.Na[0],
            'spec', self.SIGMA, self.SEED)
        Cij[:NE, :NE] = Cee * self.Jab[0][0]
        del Cee

        if self.Na[1]>0:
            Cie = generate_Cab(self.Ka[0], self.Na[1], self.Na[0],
                'None', self.SIGMA, self.SEED)
            Cij[NE:, :NE] = Cie * self.Jab[1][0]
            del Cie
        
            Cei = generate_Cab(self.Ka[1], self.Na[0], self.Na[1],
                'None', 1.0, self.SEED, PHASE=np.pi)
            Cij[:NE, NE:] = Cei * self.Jab[0][1]
            del Cei
        
            Cii = generate_Cab(self.Ka[1], self.Na[1], self.Na[1],
                'None', 1.0, self.SEED)
            Cij[NE:, NE:] = Cii * self.Jab[1][1]
            del Cii
        
        Cij = Cij.astype(np.float32)

        return Cij
    
    def run(self):
        NE = self.Na[0]        
        Cij = self.generate_Cij()
        
        self.print_params()
        
        running_step = 0
        data = []
        
        for step in tqdm(range(self.N_STEPS)):
            self.perturb_inputs(step)
            # self.update_inputs(Cij)            
            self.inputs = numba_update_inputs(Cij, self.rates, self.inputs, self.Na, self.EXP_DT_TAU_SYN)
            self.update_rates()
            # self.rates = numba_update_rates(self.rates, self.inputs, self.ff_inputs, self.Na, self.TF_NAME)
            
            running_step += 1
            
            if step >= self.N_STEADY:
                time = step * self.ones_vec
                
                if running_step >= self.N_WINDOW:
                    data.append(np.vstack((time, self.rates * 1000, self.ff_inputs, self.inputs)).T)
                    
                    print('time (ms)', np.round(step/self.N_STEPS, 2),
                          'rates (Hz)', np.round(np.mean(self.rates[:NE]) * 1000, 2),
                          np.round(np.mean(self.rates[NE:]) * 1000, 2))
                    running_step = 0

        del Cij
        data = np.stack(np.array(data), axis=0)
        self.df = nd_numpy_to_nested(data)
        
        if self.SAVE:                        
            print('saving data to', self.FILE_NAME + '_' + str(self.GAIN) + '.h5')
            store = HDFStore(self.FILE_NAME  + '_' + str(self.GAIN) + '.h5', 'w')
            store.append('data', self.df, format='table', data_columns=True)
            store.close()
        
        return self


if __name__ == "__main__":

    config = safe_load(open("./config.yml", "r"))
    model = Network(**config)

    start = perf_counter()
    model.run()
    end = perf_counter()

    print("Elapsed (with compilation) = {}s".format((end - start)))
