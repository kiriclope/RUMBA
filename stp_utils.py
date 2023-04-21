import numpy as np
from numba import jit


class STP_Model():

    def __init__(self, N, DT):

        self.USE = 0.05
        self.TAU_REC = 20.0
        self.TAU_FAC = 45.0
        
        self.u_stp = np.ones((N,), dtype=np.float32)
        self.x_stp = np.ones((N,), dtype=np.float32)

        u_plus = self.u_stp + self.USE * (1.0 - self.u_stp)
        self.A_u_x_stp = u_plus * self.x_stp
        # self.A_u_x_stp = np.zeros((N,), dtype=np.float32)
        
        self.DT = DT
        self.DT_TAU_REC = DT / self.TAU_REC
        self.DT_TAU_FAC = DT / self.TAU_FAC
        
    def markram_stp(self, rates):

        u_plus = self.u_stp + self.USE * (1.0 - self.u_stp)
        
        self.x_stp = self.x_stp + (1.0 - self.x_stp) * self.DT_TAU_REC - self.DT * u_plus * self.x_stp * rates
        
        self.u_stp = self.u_stp - self.DT_TAU_FAC * self.u_stp + self.DT * self.USE * (1.0 - self.u_stp) * rates
    
        self.A_u_x_stp = u_plus * self.x_stp

        # self.A_u_x_stp, self.u_stp, self.x_stp = numba_markram_stp(rates, self.u_stp, self.x_stp, self.USE, self.DT_TAU_REC, self.DT_TAU_FAC, self.DT)

    def hansel_stp(self, rates):
        self.x_stp = self.x_stp - self.DT_TAU_REC * (self.x_stp - 1.0) - self.DT * self.x_stp * self.u_stp * rates
        self.u_stp = self.u_stp - self.DT_TAU_FAC * (self.u_stp - self.USE) + self.DT * self.USE * rates
        
        self.A_u_x_stp = self.u_stp * self.x_stp
        
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_markram_stp(rates, u_stp, x_stp, USE, DT_TAU_REC, DT_TAU_FAC, DT):

    u_plus = u_stp + USE * (1.0 - u_stp)

    x_stp = x_stp + (1.0 - x_stp) * DT_TAU_REC - DT * u_plus * x_stp * rates
    
    u_stp = u_stp - DT_TAU_FAC * u_stp + DT * USE * (1.0 - u_stp) * rates
    
    A_u_x_stp = u_plus * x_stp

    return A_u_x_stp, u_stp, x_stp
