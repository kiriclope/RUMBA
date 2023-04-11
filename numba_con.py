import numpy as np
from numba import jit

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_Cij(Cij, rates, ALPHA=1, ETA_DT=0.01):
    # norm = np.where(Cij, ALPHA * Cij * rates**2, 0)
    # Cij = np.where(Cij, (Cij + ETA_DT * (np.outer(rates, rates) - norm)), 0)

    norm = np.where(Cij, ALPHA * Cij * rates**2, 0)
    Cij = Cij + ETA_DT * (np.outer(rates, rates) - norm)
    
    # Cij = np.where(Cij, Cij + ETA_DT * np.outer(rates, rates), 0)
    return Cij

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
