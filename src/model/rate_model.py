'''This script sets up and runs a network model. The model contains
adjustable parameters such as model connectivity (Cij), synaptic
dynamics (SYN_DYN), feedforward dynamics (FF_DYN), and other
processing details (perturbation functions, threshold dynamics, etc.)
that are user-configurable via a configuration file (conf_file). The
outputs can be saved in an HDF5 file.'''

import sys
import os
from time import perf_counter

import numpy as np
from tqdm import tqdm
from yaml import safe_load
from pandas import HDFStore

from src.model.numba_utils import (
    numba_update_ff_inputs,
    numba_update_inputs,
    numba_update_rates,
)

from src.model.utils import nd_numpy_to_nested
from src.model.connectivity import numba_generate_Cab
from src.model.plasticity import STP_Model

from src.analysis.decode import decode_bump
from src.model.mean_field_spec import get_mf_spec, m0_func


def pertur_func(theta, I0, SIGMA0, PHI0):
    '''Stimulus shape'''
    res = I0 * (1.0 + SIGMA0 * np.cos(theta - PHI0))
    return res * (res > 0)


class Network:
    '''Network class represents the overarching model that will be simulated. 
    The model parameters are derived from a given configuration file and are visualized and adjusted at runtime.'''

    def convert_times_to_steps(self):
        self.N_STEPS = int(self.DURATION / self.DT)

        self.N_STEADY = int(self.T_STEADY / self.DT)
        self.N_WINDOW = int(self.T_WINDOW / self.DT)
        
        self.N_STIM_ON = int(self.T_STIM_ON / self.DT)
        self.N_STIM_OFF = int(self.T_STIM_OFF / self.DT)
        
        self.N_CUE_ON = int((self.T_STIM_ON + 1.5) / self.DT)
        self.N_CUE_OFF = int((self.T_STIM_OFF + 1.5) / self.DT)
        
    def set_integration_const(self):
        self.EXP_DT_TAU_SYN = []
        self.DT_TAU_SYN = []
        
        if self.IF_NMDA:
            self.EXP_DT_TAU_NMDA = []
            self.DT_TAU_NMDA = []
        
        self.EXP_DT_TAU_FF = []
        self.DT_TAU_FF = []

        self.EXP_DT_TAU_MEM = []
        self.DT_TAU_MEM = []

        self.EXP_DT_TAU_THRESH = []
        self.DT_TAU_THRESH = []

        for i_pop in range(self.N_POP):
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
        
    def __init__(self, conf_file, sim_name, conf_path="/home/leon/models/rnn_numba/conf/", **kwargs):
        '''Initialize the Network model with configuration file, simulation name and
        potentially other unspecified keyword arguments.'''
        
        conf_path = conf_path + conf_file
        print('Loading config from', conf_path)
        param = safe_load(open(conf_path, "r"))

        param["FILE_NAME"] = sim_name
        param.update(kwargs)

        for k, v in param.items():
            setattr(self, k, v)

        self.FILE_PATH = self.DATA_PATH + "/" + self.FILE_NAME + ".h5"
        print('Saving to', self.FILE_PATH)

        if not os.path.exists(self.DATA_PATH):
            os.makedirs(self.DATA_PATH)
            
        if not os.path.exists(self.MAT_PATH):
            os.makedirs(self.MAT_PATH)

        self.PHASE = self.PHASE * np.pi / 180.0

        self.convert_times_to_steps()
        self.set_integration_const()
        
        # PARAMETERS
        self.N_POP = int(self.N_POP)
        self.N = int(self.N)
        # self.K = const.K

        self.ones_vec = (
            np.ones((self.N,), dtype=np.float64) / self.N_STEPS * self.DURATION
        )

        self.VAR_FF = np.array(self.VAR_FF, dtype=np.float64)

        self.Na = []
        self.Ka = []
        
        for i_pop in range(self.N_POP):
            self.Na.append(int(self.N * self.frac[i_pop]))
            # self.Ka.append(self.K * const.frac[i_pop])
            self.Ka.append(self.K)
        
        self.Na = np.array(self.Na, dtype=np.int64)
        self.Ka = np.array(self.Ka, dtype=np.float64)
        self.csumNa = np.concatenate(([0], np.cumsum(self.Na)))
        
        self.THRESH = np.array(self.THRESH, dtype=np.float64)        
        self.Jab = np.array(self.Jab, dtype=np.float64).reshape(self.N_POP, self.N_POP)

        if self.VERBOSE:
            print("Jab", self.Jab)
        
        self.STRUCTURE = np.array(self.STRUCTURE).reshape(self.N_POP, self.N_POP)
        self.SIGMA = np.array(self.SIGMA, dtype=np.float64).reshape(
            self.N_POP, self.N_POP
        )
        self.KAPPA = np.array(self.KAPPA, dtype=np.float64).reshape(
            self.N_POP, self.N_POP
        )

        if self.VERBOSE:
            print("SIGMA", self.SIGMA)
            print("KAPPA", self.KAPPA)
        
        self.SIGMA = self.SIGMA / np.abs(self.Jab)
        self.Jab *= self.GAIN
        
        if self.IF_NMDA:
            self.Jab_NMDA = (
                0.1 * self.Jab[:, 0] * self.DT_TAU_NMDA[i_pop] / np.sqrt(self.Ka[0])
            )
        
        self.Iext = np.array(self.Iext, dtype=np.float64)

        if self.VERBOSE:
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

        self.I0 = np.array(self.I0, dtype=np.float64)
        self.I0 *= self.M0  # * np.sqrt(self.Ka[0])

        self.I1 = np.random.normal(self.I1, np.sqrt(2.0 * np.array(self.I1)))
        self.I1 *= self.M0  # * np.sqrt(self.Ka[0])

        if self.PHI0 == "None":
            self.PHI0 = np.random.uniform(2 * np.pi)
        else:
            self.PHI0 = self.PHI0 * np.pi / 180.0

        self.PHI1 = self.PHI0 - self.DPHI * np.pi
        
        self.SEED = self.SEED
        if self.SEED == "None":
            self.SEED = None
        
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
        '''Print the parameters of the Network.'''
        print("Parameters:")
        print("N", self.N, "Na", self.Na, end=" ")
        print("K", self.K, "Ka", self.Ka)
        
        print("Iext", self.Iext, end=" ")
        print("Jab", self.Jab.flatten())
        print("KAPPA", self.KAPPA, "SIGMA", self.SIGMA)
        print("MF Rates:", self.mf_rates)

    def update_thresh(self):
        '''Update the threshold values based on available dynamics.'''
        for i_pop in range(self.N_POP):
            self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (
                self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                * self.EXP_DT_TAU_THRESH[i_pop]
            )
            self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]] = (
                self.thresh[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
                + self.DT_TAU_THRESH[i_pop]
                * self.THRESH[i_pop]
                * self.rates[self.csumNa[i_pop] : self.csumNa[i_pop + 1]]
            )

    def perturb_inputs(self, step):
        '''Perturb the inputs based on the simulus parameters.''' 
        if step == 0:
            self.ff_inputs_0[self.csumNa[0] : self.csumNa[0 + 1]] = 0.0

        if step == self.N_STIM_ON:
            self.ff_inputs_0[self.csumNa[0] : self.csumNa[1]] = self.Iext[0]

        if step == self.N_STIM_ON:
            if self.VERBOSE:
                print("STIM ON")
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop])
                self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] = self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] + pertur_func(
                    theta, self.I0[i_pop], self.SIGMA0, self.PHI0)

        if step == self.N_STIM_OFF:
            if self.VERBOSE:
                print("STIM OFF")
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop])
                self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] = self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] - pertur_func(
                    theta, self.I0[i_pop], self.SIGMA0, self.PHI0)

        if step == self.N_STIM_OFF:
            self.ff_inputs_0[self.csumNa[0] : self.csumNa[1]] = self.Iext[0]

        if step == self.N_CUE_ON:
            if self.VERBOSE:
                print("CUE ON")
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop])
                self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] = self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] + pertur_func(
                    theta, self.I1[i_pop], self.SIGMA0, self.PHI1)

        if step == self.N_CUE_OFF:
            if self.VERBOSE:
                print("CUE OFF")
            for i_pop in range(self.N_POP):
                theta = np.linspace(0.0, 2.0 * np.pi, self.Na[i_pop])
                self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] = self.ff_inputs_0[
                    self.csumNa[i_pop] : self.csumNa[i_pop + 1]
                ] - pertur_func(
                    theta, self.I1[i_pop], self.SIGMA0, self.PHI1)

    def generate_Cij(self):
        ''' Generate the connectivity matrix Cij according to the specific regulation rules.'''
        np.random.seed(self.SEED)
        
        Cij = np.zeros((self.N, self.N), dtype=np.float64)

        for i_post in range(self.N_POP):
            for j_pre in range(self.N_POP):
                Cab = numba_generate_Cab(
                    self.Ka[j_pre],
                    self.Na[i_post],
                    self.Na[j_pre],
                    self.STRUCTURE[i_post, j_pre],
                    self.SIGMA[i_post, j_pre],
                    self.KAPPA[i_post, j_pre],
                    self.SEED,
                    self.PHASE,
                    self.VERBOSE,
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
        '''Modify the connectivity matrix when NMDA receptors are active.'''
        Cij_NMDA = np.zeros((self.N, self.Na[0]), dtype=np.float64)

        for i_post in range(self.N_POP):
            Cij_NMDA[self.csumNa[i_post] : self.csumNa[i_post + 1]] = (
                Cij[self.csumNa[i_post] : self.csumNa[i_post + 1].copy(), : self.Na[0]]
                != 0
            ) * self.Jab_NMDA[i_post]

        return Cij_NMDA

    def stack_arrays(self, times, stp=None):
        if self.IF_STP:
            if self.N_POP == 2:
                data_stack = np.vstack(
                    (
                        times,
                        self.rates,
                        self.ff_inputs,
                        self.inputs,
                        np.pad(stp.u_stp, (0, self.Na[1])),
                        np.pad(stp.x_stp, (0, self.Na[1])),
                        np.pad(stp.A_u_x_stp, (0, self.Na[1])),
                    )
                ).T
                
            else:
                data_stack = np.vstack(
                        (
                            times,
                            self.rates,
                            self.ff_inputs,
                            self.inputs,
                            stp.u_stp,
                            stp.x_stp,
                            stp.A_u_x_stp,
                        )
                ).T
                
        else:
            data_stack = np.vstack((times, self.rates, self.ff_inputs, self.inputs)).T
            
        return data_stack

    def print_tuning(self):

        amplitudes = []
        phases = []
        
        m0, m1, phase = decode_bump(
            self.rates[:self.csumNa[1]]
        )
        
        amplitudes.append(m1)
        phases.append(phase * 180.0 / np.pi)        
        
        if self.N_POP > 1:
            m0, m1, phase = decode_bump(
                self.rates[self.csumNa[1] : self.csumNa[2]]
            )
            
            amplitudes.append(m1)
            phases.append(phase * 180.0 / np.pi)
            
        if self.N_POP > 2:
            m0, m1, phase = decode_bump(self.rates[self.csumNa[2]:])
            amplitudes.append(m1)
            phases.append(phase * 180.0 / np.pi)

        print(
            "m1", np.round(amplitudes, 2),
            "phase", np.round(phases, 2),
            flush=True,                            
        )


    def print_activity(self, times):
        
        activity = []

        activity.append(np.round(np.mean(self.rates[:self.csumNa[1]]), 2))

        if self.N_POP > 1:
            activity.append(np.round(np.mean(self.rates[self.csumNa[1]:self.csumNa[2]]), 2))

        if self.N_POP > 2:
            activity.append(np.round(np.mean(self.rates[self.csumNa[2]:]), 2))
            
        print(
            "times (s)",
            np.round(times, 2),
            "rates (Hz)",
            activity,
        )
        
    def run(self):
        '''Perform the simulation for a series of steps within the specified duration.'''
        start = perf_counter()
            
        if self.IF_LOAD_MAT:
            print('Loading matrix from', self.MAT_PATH + "/Cij.npy")
            Cij = np.load(self.MAT_PATH + "/Cij.npy")
        else:
            print('Generating matrix Cij')
            Cij = self.generate_Cij()
            
            if self.IF_SAVE_MAT:
                print('Saving matrix to', self.MAT_PATH + "/Cij.npy")
                np.save(self.MAT_PATH + "/Cij.npy", Cij)
        
        if self.IF_NMDA:
            Cij_NMDA = np.ascontiguousarray(self.generate_Cij_NMDA(Cij))
        
        np.random.seed(None)
        
        if self.IF_STP:
            stp = STP_Model(self.Na[0], self.DT)
            if self.VERBOSE:
                print("stp:", stp.USE, stp.TAU_REC, stp.TAU_FAC)
        
        if self.VERBOSE:
            self.print_params()

        running_step = 0
        data = []

        self.mean_rates = self.rates

        print('Running simulation')
        for step in range(self.N_STEPS):
        # for step in tqdm(range(self.N_STEPS)):
            self.perturb_inputs(step)

            self.ff_inputs = numba_update_ff_inputs(
                self.ff_inputs,
                self.ff_inputs_0,
                self.EXP_DT_TAU_FF,
                self.DT_TAU_FF,
                self.VAR_FF,
                self.FF_DYN,
            )

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

            self.rates = numba_update_rates(
                self.rates,
                self.ff_inputs,
                self.inputs,
                self.inputs_NMDA,
                self.thresh,
                self.csumNa,
                self.EXP_DT_TAU_MEM,
                self.DT_TAU_MEM,
                RATE_DYN=self.RATE_DYN,
                IF_NMDA=self.IF_NMDA,
            )
            
            if self.THRESH_DYN:
                self.update_thresh()
            
            running_step += 1

            if step >= self.N_STEADY:
                time_vec = (step - self.N_STEADY) * self.ones_vec
                times = np.round((step -self.N_STEADY) / self.N_STEPS * self.DURATION, 2)

                if running_step % self.N_WINDOW == 0:
                    data_stack = self.stack_arrays(time_vec)
                    data.append(data_stack)
                                        
                    if self.VERBOSE:
                        self.print_activity(times)
                        if 'cos' in self.STRUCTURE:         
                            self.print_tuning()
                        
                    running_step = 0

        del Cij
        data = np.stack(np.array(data), axis=0)
        self.df = nd_numpy_to_nested(data, N_POP=self.N_POP, IF_STP=self.IF_STP)

        if self.SAVE_DATA:
            if self.VERBOSE:
                print("saving data to", self.FILE_PATH)
            store = HDFStore(self.FILE_PATH, "w")
            store.append("data", self.df, format="table", data_columns=True)
            store.close()
        
        end = perf_counter()
        print("Elapsed (with compilation) = {}s".format((end - start)))
        

if __name__ == "__main__":
    '''If run directly, use the first command-line argument as the configuration file, and the second command-line 
    argument as the simulation name.'''
    
    conf_file = sys.argv[1]
    sim_name = sys.argv[2]
    model = Network(conf_file, sim_name)
    
    start = perf_counter()
    model.run()
    end = perf_counter()

    print("Elapsed (with compilation) = {}s".format((end - start)))

