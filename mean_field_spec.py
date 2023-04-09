import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load
from scipy.integrate import quad_vec, quad, quadrature
from scipy.special import erf, erfc
from scipy.optimize import root


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def u_theta(theta, u0, u1):
    return u0 + u1 * np.cos(theta)


def quench_avg_Phi(u, alpha):
    # Sigmoid
    # return 0.5 * erfc((1.0-u) / np.sqrt(2.0 * np.abs(alpha))) * (alpha>0)
    # threshold linear
    if alpha > 0:
        return ( 0.5 * u * erf(u / np.sqrt(2.0 * alpha) + 1.0) + np.sqrt(alpha / 2.0 / np.pi) * np.exp((-u**2)/(2.0 * alpha)) )
    else:
        return np.nan


def integrand(theta, u0, u1, alpha):
    return quench_avg_Phi(u_theta(theta, u0, u1), alpha)


def integrand2(theta, u0, u1, alpha):
    return quench_avg_Phi(u_theta(theta, u0, u1), alpha) * np.cos(theta)


def m0_func(u0, u1, alpha):
    if alpha <= 0 :
        res = np.nan
    else:
        res, err = quad(integrand, 0, 2.0 * np.pi, args=(u0, u1, alpha), limit=50)
    # res, err = quadrature(integrand, 0, np.pi, args=(u0, u1, alpha), miniter=40)
    # res, err = quad_vec(integrand, 0, np.pi, args=(u0, u1, alpha), workers=-1)
    return res / 2.0 / np.pi


def m1_func(u0, u1, alpha):
    if alpha <= 0:
        res = np.nan
    else:
        res, err = quad(integrand2, 0, 2.0 * np.pi, args=(u0, u1, alpha), limit=50)
    # res, err = quadrature(integrand2, 0, np.pi, args=(u0, u1, alpha), miniter=40)
    # res, err = quad_vec(integrand2, 0, np.pi, args=(u0, u1, alpha), workers=-1)
    return res * 1.0 / np.pi


m0_func = np.vectorize(m0_func)
m1_func = np.vectorize(m1_func)

class MeanFieldSpec:

    def __init__(self, **kwargs):
        const = Bunch(kwargs)

        self.verbose = const.verbose
        self.TOLERANCE = const.TOLERANCE
        self.MAXITER = const.MAXITER

        # PARAMETERS
        self.N_POP = int(const.N_POP)
        self.N = int(const.N)
        self.K = const.K

        if self.K == 'Inf':
            self.K = np.inf
            
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

        print('kappa', self.kappa.flatten())
        self.kappa_Jab = self.Jab * self.kappa
        self.Jab2 = self.Jab * self.Jab
        # print('Jab2', self.Jab2)

        self.Iext *= const.M0
        
        # print('kappa_Jab', self.kappa_Jab)
        try:            
            self.mf_rates = -np.dot(np.linalg.inv(self.Jab), self.Iext)
            print('MF Rates:', self.mf_rates)
        except:
            pass
        
        self.initial_guess = np.random.rand(3* self.N_POP)

    def self_consistent_eqs(self, x):

        u0 = x[:self.N_POP] # mean input
        u1 = x[self.N_POP:2*self.N_POP] # first fourier moment of the input
        alpha = np.abs(x[2*self.N_POP:3*self.N_POP]) # variance
        
        m0 = m0_func(u0, u1, alpha)
        u0_eq = u0 / np.sqrt(self.K) - (self.Iext + np.dot(self.Jab, m0))
        
        m1 = m1_func(u0, u1, alpha)
        u1_eq = u1 - np.dot(self.kappa_Jab, m1)
        
        alpha_eq = alpha - np.dot(self.Jab2, m0**2)
        
        eqs = np.array([u0_eq, u1_eq, alpha_eq])

        return eqs.flatten()

    def solve(self):
        self.solutions()
        if self.verbose:
            print('initial guess', 'm0', np.round(self.m0, 3), 'm1', np.round(self.m1, 3))
        self.error = self.self_consistent_eqs(self.initial_guess)
        # print('initial error', self.error)

        counter=0
        while any( self.error > self.TOLERANCE ):
            
            self.initial_guess = np.random.rand(3 * self.N_POP) * 2.0 - 1.0
            self.result = root(self.self_consistent_eqs, self.initial_guess, method='hybr', tol=self.TOLERANCE)
            self.error = self.self_consistent_eqs(self.result.x)
            self.solutions()

            if any(np.isnan(self.m0)):
                self.error = np.ones(3)
            
            if self.verbose:
                print('iter', counter, 'm0', np.round(self.m0, 3), 'm1', np.round(self.m1, 3))
                print('error', self.error)

            if counter >= self.MAXITER :
                print('ERROR: max number of iterations reached')
                # print('error', self.error > self.TOLERANCE)
                break

            counter+=1

    def solutions(self):

        try:
            u0 = self.result.x[:self.N_POP] # mean input
            u1 = self.result.x[self.N_POP:2*self.N_POP] # first fourier moment
            alpha = self.result.x[2*self.N_POP:3*self.N_POP] # variance
        except:
            u0 = self.initial_guess[:self.N_POP]
            u1 = self.initial_guess[self.N_POP:2*self.N_POP]
            alpha = self.initial_guess[2*self.N_POP:3*self.N_POP]

        self.m0 = m0_func(u0, u1, alpha)
        self.m1 = m1_func(u0, u1, alpha)

        # print('m0', self.m0, 'm1', self.m1)

    def bifurcation(self):
        self.m0_list = []
        self.m1_list = []

        self.kappas = np.linspace(6, 8, 10)
        for kappa in self.kappas:
            self.kappa[0][0] = kappa
            # self.kappa[0][1] = 0.5 * kappa 
            # self.kappa[1][0] = kappa
            # self.kappa[1][1] = 0.5 * kappa
           
            self.kappa_Jab = self.Jab * self.kappa

            self.solve()
            self.m0_list.append(self.m0)
            self.m1_list.append(self.m1)
            
            print(self.kappa[0][0], 'm0', self.m0, 'm1', self.m1)

        self.m0_list = np.array(self.m0_list).T
        self.m1_list = np.array(self.m1_list).T
        
if __name__ == "__main__":

    config = safe_load(open("./configEE.yml", "r"))
    model = MeanFieldSpec(**config)
    model.solve()
    print('m0', model.m0, 'm1', model.m1)
    fig, ax = plt.subplots(1,2)
    for iter in range(10):
        print('iter', iter)
        model.bifurcation()
    
        ax[0].plot(model.kappas, np.abs(model.m0_list[0]), 'ro')
        # ax[0].plot(model.kappas, np.abs(model.m0_list[1]), 'ko')
        # ax[0].plot(model.kappas, np.abs(model.m0_list[2]), 'bo')
    
        ax[1].plot(model.kappas, np.abs(model.m1_list[0]), 'ro')
        # ax[1].plot(model.kappas, np.abs(model.m1_list[1]), 'ko')
        # ax[1].plot(model.kappas, np.abs(model.m1_list[2]), 'bo')

