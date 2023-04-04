import numpy as np
from time import perf_counter
from yaml import safe_load
from tqdm import tqdm
from scipy.special import erf
from pandas import DataFrame, HDFStore, concat
from numba import jit


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_Cij(Cij, rates, DT_TAU_SYN, ALPHA=1, ETA_DT=0.01):
    norm = np.where(Cij, ALPHA * Cij / DT_TAU_SYN[0] * rates**2, 0) 
    Cij = np.where(Cij, DT_TAU_SYN[0] * (Cij + ETA_DT * (np.outer(rates, rates) - norm)), 0)
    return Cij 

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

    if X.shape[-1] == 5:
        variables = ['rates', 'ff', 'h_E', 'h_I']
    else:
        variables = ['rates', 'ff', 'h_E1', 'h_E2', 'h_I']
        
    df = DataFrame()
    idx = np.arange(0, X.shape[1], 1)
    for i_time in range(X.shape[0]): 
        df_i = DataFrame(X[i_time,:, 1:], columns=variables)
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
def TF(x, thresh, tfname='TL',  gain=1.0):
    if tfname=='TL':
        return x * (x > thresh) 
    
    # if tfname=='NL':
    #     x_nthresh = x**2 * (x > 0) 
    #     x_thresh = (x>=thresh) * (np.sqrt(gain * np.abs(x)) - np.sqrt(gain * thresh) + thresh**2 - x**2)
    #     return  x_nthresh + x_thresh

    # if tfname=='Sig':
    #     return thresh / (1.0 + 1.0 * np.exp(-(x+10.0)/10))
    # elif tfname=='Sig':
    #     return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    # elif tfname=='LIF':
    #     return - 1.0 * (x > 1.0) / np.log(1.0 - 1.0 / x)


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
def generate_Cab(Kb, Na, Nb, STRUCTURE='None', SIGMA=1, SEED=None, PHASE=0):

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

    elif "weak" in STRUCTURE:
        print('with weak proba')
        Pij[:, :] = Pij[:, :] * np.float32(SIGMA) / np.sqrt(Kb) 
            
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
def numba_update_inputs(Cij, rates, inputs, Na, csumNa, EXP_DT_TAU_SYN):

    for i_pop in range(len(Na)):
        inputs[i_pop] = inputs[i_pop] * EXP_DT_TAU_SYN[i_pop] 
        inputs[i_pop] = inputs[i_pop] + np.dot(Cij[:, csumNa[i_pop]:csumNa[i_pop+1]], rates[csumNa[i_pop]:csumNa[i_pop+1]])
        
    return inputs


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_rates(rates, inputs, ff_inputs, Na, TF_NAME):
    net_inputs = ff_inputs
    
    for i_pop in range(len(Na)):
        net_inputs = net_inputs + inputs[i_pop]
    rates = TF(net_inputs, TF_NAME)
    
    return rates


class Network:
    def __init__(self, **kwargs):

        const = Bunch(kwargs)

        self.IF_LEARNING = const.IF_LEARNING
        
        self.SAVE = const.SAVE
        self.verbose = const.verbose
        if self.verbose:
            print(kwargs.keys())

        self.FILE_NAME = const.FILE_NAME
        self.TF_NAME = const.TF_NAME
        self.TF_GAIN = const.TF_GAIN
        
        self.RATE_DYN = const.RATE_DYN
        # SIMULATION
        self.DT = const.DT

        self.ALPHA = 1.0
        self.ETA_DT = self.DT / 10.0
        
        self.DURATION = float(const.DURATION)
        self.N_STEPS = int(self.DURATION / self.DT)

        self.N_STEADY = int(const.T_STEADY / self.DT)
        self.N_WINDOW = int(const.T_WINDOW / self.DT)

        self.N_STIM_ON = int(const.T_STIM_ON / self.DT)
        self.N_STIM_OFF = int(const.T_STIM_OFF / self.DT)

        # PARAMETERS
        self.N_POP = int(const.N_POP)
        self.N = int(const.N)
        self.K = const.K

        self.idx = np.arange(self.N)
        self.idx.astype(np.float32)
        self.ones_vec = np.ones((self.N,), dtype=np.float32) / self.N_STEPS

        self.TAU_SYN = const.TAU_SYN 
        self.TAU_FF = const.TAU_FF
        self.TAU_MEM = const.TAU_MEM
        
        self.Na = []
        self.Ka = []

        self.EXP_DT_TAU_SYN = []
        self.DT_TAU_SYN = []

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


        self.Na = np.array(self.Na, dtype=np.int64)
        self.Ka = np.array(self.Ka, dtype=np.float32)

        self.EXP_DT_TAU_SYN = np.array(self.EXP_DT_TAU_SYN, dtype=np.float32)
        self.DT_TAU_SYN = np.array(self.DT_TAU_SYN, dtype=np.float32)

        if self.IF_NMDA:
            self.EXP_DT_TAU_NMDA = np.array(self.EXP_DT_TAU_NMDA, dtype=np.float32)
            self.DT_TAU_NMDA = np.array(self.DT_TAU_NMDA, dtype=np.float32)            
        
        self.EXP_DT_TAU_MEM = np.array(self.EXP_DT_TAU_MEM, dtype=np.float32)
        self.DT_TAU_MEM = np.array(self.DT_TAU_MEM, dtype=np.float32)

        self.csumNa = np.concatenate(([0], np.cumsum(self.Na)))
        
        self.THRESH_DYN = const.THRESH_DYN
        self.THRESH = np.array(const.THRESH, dtype=np.float32)
        
        self.M0 = const.M0

        self.Jab = np.array(const.Jab, dtype=np.float32).reshape(self.N_POP, self.N_POP)
        
        print('Jab', self.Jab)
        
        self.GAIN = const.GAIN
        self.Jab *= const.GAIN

        if self.IF_NMDA:
            self.Jab_NMDA = self.Jab[:, 0] * self.DT_TAU_NMDA[i_pop] / np.sqrt(self.Ka[0]) 
            
        self.Iext = np.array(const.Iext, dtype=np.float32)

        print('Iext', self.Iext)

        try:
            self.mf_rates = -np.dot(np.linalg.inv(self.Jab), self.Iext)
        except:
            self.mf_rates = np.nan

        for i_pop in range(self.N_POP):
            self.Jab[:, i_pop] = self.Jab[:, i_pop] * self.DT_TAU_SYN[i_pop] / np.sqrt(self.Ka[i_pop]) 
            
        # self.Jab[-1, :] = self.Jab[-1,:] * np.sqrt(self.Ka[-1]) / self.Ka[-1]
        
        self.Jab[np.isinf(self.Jab)] = 0        
        
        self.Iext *= np.sqrt(self.Ka[0]) * self.M0

        self.I0 = np.array(const.I0, dtype=np.float32)
        self.I0 *= self.M0 * np.sqrt(self.Ka[0])
        
        self.PHI0 = np.random.uniform(2*np.pi)
        
        self.SEED = const.SEED
        if self.SEED == 'None':
            self.SEED = None

        self.STRUCTURE = np.array(const.STRUCTURE).reshape(self.N_POP, self.N_POP)
        self.SIGMA = np.array(const.SIGMA, dtype=np.float32).reshape(self.N_POP, self.N_POP)
                
        self.rates = np.ascontiguousarray(np.zeros( (self.N,), dtype=np.float32))
        self.inputs = np.zeros((self.N_POP, self.N), dtype=np.float32)

        if self.IF_NMDA:
            self.inputs_NMDA = np.zeros((self.N_POP, self.N), dtype=np.float32)

        # rng = np.random.default_rng()
        # self.rates[:self.Na[0]] = rng.normal(5, 1, self.Na[0])
        # self.rates[self.Na[0]:] = rng.normal(10, 2, self.Na[1])
        
        self.ff_inputs = np.zeros((self.N,), dtype=np.float32)
        self.ff_inputs_0 = np.ones((self.N,), dtype=np.float32)
        
        self.thresh = np.ones( (self.N,), dtype=np.float32) 

        for i_pop in range(self.N_POP):
            self.ff_inputs_0[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = self.ff_inputs_0[self.csumNa[i_pop]:self.csumNa[i_pop+1]] * self.Iext[i_pop]
            self.thresh[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = self.thresh[self.csumNa[i_pop]:self.csumNa[i_pop+1]] * self.THRESH[i_pop]
        
    def print_params(self):
        print('Parameters:')
        print('N', self.N, 'Na', self.Na)
        print('K', self.K, 'Ka', self.Ka)

        print('Iext', self.Iext)
        print('Jab', self.Jab.flatten())

        print('MF Rates:', self.mf_rates)
    
    def update_inputs(self, Cij):
        for i_pop in range(self.N_POP):
            self.inputs[i_pop] = self.inputs[i_pop] * self.EXP_DT_TAU_SYN[i_pop]    
            Cab = Cij[:, self.csumNa[i_pop]:self.csumNa[i_pop+1]]
            rb = rates[self.csumNa[i_pop]:self.csumNa[i_pop+1]]
            self.inputs[i_pop] = self.inputs[i_pop] + np.dot(Cab, rb)
    
    def update_rates(self):
        if self.FF_DYN==0:
            net_inputs = self.ff_inputs_0
        else:
            net_inputs = self.ff_inputs
        
        for i_pop in range(self.N_POP):
            net_inputs = net_inputs + self.inputs[i_pop]

        if self.IF_NMDA:
            for i_pop in range(self.N_POP):
                net_inputs = net_inputs + self.inputs_NMDA[i_pop]
            
        if self.RATE_DYN==0:
            self.rates = TF(net_inputs, self.thresh, self.TF_NAME)
        else:
            for i_pop in range(self.N_POP):
                # idx = np.arange(self.csumNa[i_pop], self.csumNa[i_pop+1], 1)
                self.rates[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = self.rates[self.csumNa[i_pop]:self.csumNa[i_pop+1]] * self.EXP_DT_TAU_MEM[i_pop] 
                self.rates[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = self.rates[self.csumNa[i_pop]:self.csumNa[i_pop+1]] + self.DT_TAU_MEM[i_pop] * TF(net_inputs[self.csumNa[i_pop]:self.csumNa[i_pop+1]], self.thresh[self.csumNa[i_pop]:self.csumNa[i_pop+1]], self.TF_NAME, self.TF_GAIN)
            
        # NE = self.Na[0]
        # net_inputs = self.ff_inputs
        
        # net_inputs = net_inputs + self.inputs[0]        
        # if self.Na[1]>0:
        #     net_inputs = net_inputs + self.inputs[1]   

        # if self.RATE_DYN==0:
        #     self.rates = TF(net_inputs, self.thresh, self.TF_NAME, self.TF_GAIN)
        # else:
        #     self.rates[:NE] = self.rates[:NE] * self.EXP_DT_TAU_MEM[0] # excitatory rates
        #     self.rates[:NE] = self.rates[:NE] + self.DT_TAU_MEM[0] * TF(net_inputs[:NE], self.thresh[:NE], self.TF_NAME, self.TF_GAIN)
        
        #     if self.Na[1]>0:
        #         self.rates[NE:] = self.rates[NE:] * self.EXP_DT_TAU_MEM[1]  # inhibitory rates
        #         self.rates[NE:] = self.rates[NE:] + self.DT_TAU_MEM[1] * TF(net_inputs[NE:], self.thresh[NE:], self.TF_NAME, self.TF_GAIN)

    def update_thresh(self):
        self.thresh = self.thresh * self.EXP_DT_TAU_MEM[0] 
        self.thresh = self.thresh + self.DT_TAU_MEM[0] * self.THRESH[0]

    def update_ff_inputs(self):
        self.ff_inputs = self.ff_inputs * self.EXP_DT_TAU_FF[0]
        self.ff_inputs = self.ff_inputs + self.DT_TAU_FF[0] * self.ff_inputs_0
    
    def perturb_inputs(self, step):
        NE = self.Na[0]
        NI = self.Na[0] + self.Na[1]
        
        if step == self.N_STIM_ON:            
            print('CUE ON')
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop]) 
                self.ff_inputs_0[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = self.ff_inputs_0[self.csumNa[i_pop]:self.csumNa[i_pop+1]] + self.I0[i_pop] * (1.0 + np.cos(theta - self.PHI0) )
            
        if step == self.N_STIM_OFF:
            print('CUE OFF')
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop]) 
                self.ff_inputs_0[self.csumNa[i_pop]:self.csumNa[i_pop+1]] = self.ff_inputs_0[self.csumNa[i_pop]:self.csumNa[i_pop+1]] - self.I0[i_pop] * (1.0 + np.cos(theta - self.PHI0) )
    
    def generate_Cij(self):
        Cij = np.zeros((self.N, self.N), dtype=np.float32)
        
        for i_post in range(self.N_POP):
            for j_pre in range(self.N_POP):
                Cab = generate_Cab(self.Ka[j_pre], self.Na[i_post], self.Na[j_pre],
                    self.STRUCTURE[i_post, j_pre], self.SIGMA[i_post, j_pre], self.SEED)
                Cij[self.csumNa[i_post]:self.csumNa[i_post+1], self.csumNa[j_pre]:self.csumNa[j_pre+1]] = Cab * self.Jab[i_post][j_pre]
        
        Cij = Cij.astype(np.float32)

        return Cij

    def generate_Cij_NMDA(self, Cij):
        Cij_NMDA = np.zeros((self.N, self.Na[0]), dtype=np.float32)
        
        for i_post in range(self.N_POP):
            Cij_NMDA[self.csumNa[i_post]:self.csumNa[i_post+1]] = (Cij[self.csumNa[i_post]:self.csumNa[i_post+1], :self.Na[0]]!=0) * self.Jab_NMDA[i_post]
        
        Cij_NMDA = Cij_NMDA.astype(np.float32)

        return Cij_NMDA

    def update_Cij(self, Cij):
        Cab = Cij[self.csumNa[0]:self.csumNa[1], self.csumNa[0]:self.csumNa[1]]
        rE = self.rates[:self.csumNa[1]]
        Cab = numba_update_Cij(Cab, rE, self.DT_TAU_SYN, ALPHA=self.ALPHA, ETA_DT=self.ETA_DT)
        Cij[self.csumNa[0]:self.csumNa[1], self.csumNa[0]:self.csumNa[1]] = Cab
        return Cij
    
    def run(self):
        NE = self.Na[0] 
        Cij = np.ascontiguousarray(self.generate_Cij())

        if self.IF_NMDA:
            Cij_NMDA = np.ascontiguousarray(self.generate_Cij_NMDA(Cij))
        
        self.print_params()
        
        running_step = 0
        data = []
        
        for step in tqdm(range(self.N_STEPS)):
            self.perturb_inputs(step)
            
            if self.FF_DYN:
                self.update_ff_inputs()
            
            self.inputs = numba_update_inputs(Cij, self.rates, self.inputs, self.Na, self.csumNa, self.EXP_DT_TAU_SYN)
            
            if self.IF_NMDA:
                self.inputs_NMDA = numba_update_inputs(Cij_NMDA, self.rates, self.inputs_NMDA, self.Na, self.csumNa, self.EXP_DT_TAU_NMDA)
            
            self.update_rates()
            if self.IF_LEARNING:
                Cij = self.update_Cij(Cij)
            # self.rates = numba_update_rates(self.rates, self.inputs, self.ff_inputs, self.Na, self.TF_NAME)

            if self.THRESH_DYN==0:
                self.update_thresh()
                
            running_step += 1
            
            if step >= self.N_STEADY:
                time = step * self.ones_vec
                
                if running_step >= self.N_WINDOW:
                    data.append(np.vstack((time, self.rates * 1000, self.ff_inputs, self.inputs)).T)
                    
                    print('time (ms)', np.round(step/self.N_STEPS, 2),
                          'rates (Hz)', np.round(np.mean(self.rates[:NE]) * 1000, 2),
                          np.round(np.mean(self.rates[NE:self.csumNa[2]]) * 1000, 2),
                          np.round(np.mean(self.rates[self.csumNa[2]:]) * 1000, 2))
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
