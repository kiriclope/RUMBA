import numpy as np
from numba import jit, njit
from scipy.special import i0

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_DJij(DJij, rates, EXP_DT_TAU, KAPPA_DT_TAU, ALPHA):
    
    # DJij = (DJij>0) * DJij * EXP_DT_TAU
    
    norm = ALPHA * DJij * rates**2 
    # DJij = DJij + KAPPA_DT_TAU *(np.outer(rates, rates) - norm)
    DJij = KAPPA_DT_TAU * (np.outer(rates, rates) - norm)
    print(np.mean(DJij))    
    
    # print(np.mean(DJij), np.mean(KAPPA_DT_TAU * np.outer(rates, rates)))
    
    return DJij

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def gaussian(theta, sigma):
    sigma = sigma * np.pi / 180.0
    
    if sigma > 0:
        return np.exp(-theta**2 / 2.0 / sigma**2) / np.sqrt(2.0 * np.pi) / sigma 
    else:
        return np.ones(theta.shape)

    
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def von_mises(theta, kappa):
    return np.exp(kappa * np.cos(theta)) / (2.0 * np.pi * i0(kappa))


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_update_Cij(Cij, rates, ALPHA=1, ETA_DT=0.01):
    norm = (Cij>0) * ALPHA * Cij * rates**2 
    Cij = Cij + ETA_DT * (np.outer(rates, rates) - norm)
    print(np.mean(Cij))
    
    return Cij


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def theta_mat(theta, phi):
    theta_mat = np.zeros((phi.shape[0], theta.shape[0]))

    twopi = np.float32(2.0 * np.pi)

    for i in range(phi.shape[0]):
        for j in range(theta.shape[0]):
            dtheta = np.abs(phi[i] - theta[j])
            theta_mat[i, j] = np.minimum(dtheta, twopi - dtheta)
    
    return theta_mat


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def strided_method(ar):
    a = np.concatenate((ar, ar[1:]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L-1:], (L, L), (-n, n))


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_i0(*args):
    return i0(*args)

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def generate_Cab(Kb, Na, Nb, STRUCTURE='None', SIGMA=1, KAPPA=0.5, SEED=None, PHASE=0):

    Pij = np.zeros((Na, Nb), dtype=np.float32)
    Cij = np.zeros((Na, Nb), dtype=np.float32)
    
    print('random connectivity')
    if STRUCTURE != 'None':
        theta = np.linspace(0.0, 2.0 * np.pi, Nb)
        phi = np.linspace(0.0, 2.0 * np.pi, Na)

        if 'perm' in STRUCTURE:
            print('permuted map')
            theta = np.random.permutation(theta)
            # phi = np.random.permutation(phi)
            
        theta = theta.astype(np.float32)
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
        Pij[:, :] = Pij[:, :] * np.float32(KAPPA)

    elif "spec_cos" in STRUCTURE:
        print('with spec cosine structure')
        Pij[:, :] = Pij[:, :] * np.float32(KAPPA) / np.sqrt(Kb)
        
    elif "gauss" in STRUCTURE:
        Pij[:, :] = gaussian(theta_ij, np.float64(SIGMA))
    
    if "all" in STRUCTURE:
        print('with all to all cosine structure')
        # Pij[:, :] = Pij[:, :] * np.float32(SIGMA) 
        # Cij[:, :] = (Pij[:, :] + 1.0)
        
        # Cij[:, :] = Cij[:, :] * (Cij>=0)
        
        # Z = Nb / np.sum(Cij, axis=1) / (2.0 * np.pi)
        # Cij[:, :] = Cij[:, :] * Z
        
        # Cij[:, :] = gaussian(theta_ij, np.float64(SIGMA))
        # Z = Nb / np.sum(Cij, axis=1)
        # Cij[:, :] = Cij[:, :] * Z
        
        Cij[:, :] = von_mises(theta_ij, np.float64(KAPPA))

    elif STRUCTURE == "gauss":
        print('with strong gauss proba') 
        Z = Kb / np.sum(Pij, axis=1) 
        Pij[:, :] = Pij[:, :] * Z 
        Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Pij)
        
    elif "spec_gauss" in STRUCTURE:
        print('with spec gauss proba') 
        # Z = KAPPA * np.sqrt(Kb) / np.sum(Pij, axis=1)
        # Pij[:, :] = Kb / Nb + Pij[:, :] * Z        

        # Pij[:, :] = 1.0 + KAPPA / np.sqrt(Kb) * Pij[:, :]
        # Z = Kb / np.sum(Pij, axis=1) 
        # Pij[:, :] = Pij[:, :] * Z
        
        # Z = KAPPA * np.sqrt(Kb) / np.sum(Pij, axis=1) 
        # Pij[:, :] = (Kb - KAPPA * np.sqrt(Kb)) / Nb + Pij[:, :] * Z

        # Z = KAPPA * np.sqrt(Kb) / np.sum(Pij, axis=1) 
        # Pij[:, :] = (Kb - KAPPA * np.sqrt(Kb)) / Nb + Pij[:, :] * Z

        # Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Pij)
        
        Z = KAPPA * np.sqrt(Kb) / np.sum(Pij, axis=1) 
        Pij[:, :] = Pij[:, :] * Z 
        
        Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < (Kb - KAPPA * np.sqrt(Kb)) /Nb) + 1.0 * (np.random.rand(Na, Nb) < Pij)

    elif "spec_rand" in STRUCTURE:
        print("with random spec")
        Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Kb / Nb) + KAPPA * (np.random.rand(Na, Nb) < 2.0 * np.sqrt(Kb) / Nb)
        
        # Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < (Kb - KAPPA * np.sqrt(Kb)) /Nb) + 1.0 * (np.random.rand(Na, Nb) < KAPPA * np.sqrt(Kb) /Nb)

    elif "spec_cos_rand" in STRUCTURE:
        print("with random cos spec")
        Pij[:, :] = np.sqrt(Kb) * Pij[:, :] + 1.0
        Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Kb / Nb) + SIGMA * (np.random.rand(Na, Nb) < (2.0 * np.sqrt(Kb) / Nb) * Pij )
        
    else:        
        Pij[:, :] = Pij[:, :] + 1.0
        Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < (Kb / Nb) * Pij)
    
    return Cij
