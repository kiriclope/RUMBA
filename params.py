import numpy as np

DT = 0.1  # in ms
DURATION = 1  # in ms

N = 1000
NE = 0.5 * N

VL = [0, 0]
V_TH = 1

TAU = [3, 2]
TAU_MEM = [10, 10]

EXP_DT_TAU = [np.exp(-DT/TAU[0]), np.exp(-DT/TAU[1])]
EXP_DT_TAU_MEM = [np.exp(-DT/TAU_MEM[0]), np.exp(-DT/TAU_MEM[1])]

DT_TAU = [DT/TAU_MEM[0], DT/TAU_MEM[1]]

J = np.zeros((2,2))
