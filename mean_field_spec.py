import numpy as np
from yaml import safe_load
from scipy.integrate import quad_vec
from scipy.special import erfc
from scipy.optimize import root


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def u_theta(theta, u0, u1):
    return u0 + u1 * np.cos(2.0 * theta)


def quench_avg_Phi(u, alpha):
    return 0.5 * erfc((1.0-u) / np.sqrt(2.0 * np.abs(alpha)))


def integrand(theta, u0, u1, alpha):
    return quench_avg_Phi(u_theta(theta, u0, u1), alpha) / np.pi


def integrand2(theta, u0, u1, alpha):
    return quench_avg_Phi(u_theta(theta, u0, u1), alpha) * np.cos(2.0 * theta) * 2.0 / np.pi


def m0_func(u0, u1, alpha):
    return quad_vec(integrand, 0, np.pi, args=(u0, u1, alpha), workers=-1)


def m1_func(u0, u1, alpha):
    return quad_vec(integrand2, 0, np.pi, args=(u0, u1, alpha), workers=-1)


class MeanFieldSpec:

    def __init__(self, **kwargs):
        const = Bunch(kwargs)

        self.TOLERANCE = 1e-3
        self.MAXITER = 100

        # PARAMETERS
        self.N_POP = int(const.N_POP)
        self.N = int(const.N)
        self.K = const.K

        self.Na = []
        self.Ka = []

        for i_pop in range(self.N_POP):
            self.Na.append(int(self.N * const.frac[i_pop]))
            # self.Ka.append(self.K * const.frac[i_pop])
            self.Ka.append(self.K)

        self.Na = np.array(self.Na, dtype=np.int64)
        self.Ka = np.array(self.Ka, dtype=np.float32)

        self.csumNa = np.concatenate(([0], np.cumsum(self.Na)))

        self.Jab = np.array(const.Jab, dtype=np.float32).reshape(self.N_POP, self.N_POP)
        print('Jab', self.Jab)
        self.Iext = np.array(const.Iext, dtype=np.float32)
        print('Iext', self.Iext)

        self.Jab *= const.GAIN
        self.Jab[np.isinf(self.Jab)] = 0

        self.kappa = np.array(const.SIGMA, dtype=np.float32).reshape(self.N_POP, self.N_POP)

        self.kappa_Jab = self.Jab * self.kappa
        self.Jab2 = self.Jab * self.Jab
        # print('Jab2', self.Jab2)
        
        self.Iext *= const.M0

        # print('kappa_Jab', self.kappa_Jab)
        self.mf_rates = -np.dot(np.linalg.inv(self.Jab), self.Iext)
        print('MF Rates:', self.mf_rates)
        
        self.initial_guess = np.random.rand(3* self.N_POP)

    def self_consistent_eqs(self, x):

        u0 = x[:self.N_POP] # mean input
        u1 = x[self.N_POP:2*self.N_POP] # first fourier moment of the input
        alpha = np.abs(x[2*self.N_POP:3*self.N_POP]) # variance

        m0 = m0_func(u0, u1, alpha)[0]
        u0_eq = u0 / np.sqrt(self.K) - (self.Iext + np.dot(self.Jab, m0)) 

        m1 = m1_func(u0, u1, alpha)[0]
        u1_eq = u1 - np.dot(self.kappa_Jab, m1)
        
        alpha_eq = alpha - np.abs(np.dot(self.Jab2, m0))

        eqs = np.array([u0_eq, u1_eq, alpha_eq])
        
        return eqs.flatten()


    def solve(self):
        print('initial guess', self.initial_guess)

        u0 = self.initial_guess[:self.N_POP] # mean input
        u1 = self.initial_guess[self.N_POP:2*self.N_POP] # first fourier moment
        alpha = self.initial_guess[2*self.N_POP:3*self.N_POP] # variance
        
        m0 = m0_func(u0, u1, alpha)[0]
        m1 = m1_func(u0, u1, alpha)[0]
        
        print('m0', m0, 'm1', m1)
        
        self.error = self.self_consistent_eqs(self.initial_guess)
        print('initial error', self.error)

        counter=0
        while any( self.error > self.TOLERANCE ):
            self.initial_guess = np.random.rand(3 * self.N_POP)
            self.result = root(self.self_consistent_eqs, self.initial_guess, method='lm', tol=self.TOLERANCE)
            self.error = self.self_consistent_eqs(self.initial_guess)

            if counter >= self.MAXITER :
                print('ERROR: max number of iterations reached')
                print('error', self.error)
                break

            counter+=1

        self.solutions()

    def solutions(self):

        u0 = self.result.x[:self.N_POP] # mean input
        u1 = self.result.x[self.N_POP:2*self.N_POP] # first fourier moment
        alpha = self.result.x[2*self.N_POP:3*self.N_POP] # variance
        
        m0 = m0_func(u0, u1, alpha)[0]
        m1 = m1_func(u0, u1, alpha)[0]

        print('m0', m0, 'm1', m1)

if __name__ == "__main__":

    config = safe_load(open("./config.yml", "r"))
    model = MeanFieldSpec(**config)
    model.solve()
