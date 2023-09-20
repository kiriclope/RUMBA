import numpy as np
from numba import jit

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_normal(size, SEED=0):
    """
    Initialize a NumPy array with normally distributed random numbers.

    Parameters
    ----------
    size : tuple
        Shape of the array to be initialized.
    SEED : int, optional
        Random seed, defaults to 0.

    Returns
    -------
    res : np.array
        Array filled with random numbers.
    """
    
    np.random.seed(SEED)

    res = np.zeros(size)
    if len(size)==2:
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i,j] = np.random.standard_normal()
                # res[i,j] = np.random.uniform(0.0, 1.0)
    else:
        for i in range(res.shape[0]):
            # res[i] = np.random.standard_normal()
            res[i] = np.random.uniform(0.0, 1.0)

    return res


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def theta_mat(theta, phi):
    """
    Calculate a matrix based on the difference between each pair of elements in two given arrays.

    Parameters
    ----------
    theta, phi : np.array
        Input arrays.

    Returns
    -------
    theta_mat : np.array
        Matrix based on the absolute difference between each pair of elements in 'theta' and 'phi'.
    """
    
    theta_mat = np.zeros((phi.shape[0], theta.shape[0]))

    twopi = np.float64(2.0 * np.pi)

    for i in range(phi.shape[0]):
        for j in range(theta.shape[0]):
            # theta_mat[i, j] = phi[i] - theta[j]
            dtheta = np.abs(phi[i] - theta[j])
            theta_mat[i, j] = np.minimum(dtheta, twopi - dtheta)

    return theta_mat


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def strided_method(ar):
    """
    Apply striding method on given array.

    Parameters
    ----------
    ar : np.array
        Input array.

    Returns
    -------
    np.array
        Strided array.
    """
    
    a = np.concatenate((ar, ar[1:]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L-1:], (L, L), (-n, n))


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_generate_Cab(Kb, Na, Nb, STRUCTURE='None', SIGMA=1.0, KAPPA=0.5, SEED=0, PHASE=0, verbose=0):
    """
    Generate matrix Cij based on given parameters.

    Parameters
    ----------
    Kb, Na, Nb : int
        Kb, number of presynaptic inputs, Na number of postsynaptic neurons and Nb number of presynaptic neurons
    STRUCTURE : str, optional
        Determines the network structure.
    SIGMA, KAPPA : float, optional
        Modifiers for the structure calculation.
    SEED : int, optional
        Random seed for use in the numpy random function.
    PHASE : int, optional
        Phase shift for when 'STRUCTURE' includes cosine.
    verbose : int, optional
        Controls print statements for debugging purposes.

    Returns
    -------
    Cij : np.array
        Generated matrix Cij.
    """

    # np.random.seed(SEED)

    # Pij = np.zeros((Na, Nb), dtype=np.float64)
    Cij = np.zeros((Na, Nb), dtype=np.float64)
    
    if 'cos' in STRUCTURE:
        
        Pij = np.zeros((Na, Nb), dtype=np.float64)
        
        theta = np.linspace(0.0, 2.0 * np.pi, Nb)
        phi = np.linspace(0.0, 2.0 * np.pi, Na)        

        theta_ij = theta_mat(theta, phi)
        cos_ij = np.cos(theta_ij - PHASE)
        
        Pij[:, :] = cos_ij
        Pij[:, :] = Pij * KAPPA
        
    if "all" in STRUCTURE:
        if verbose:
            print('all to all connectivity')
        # itskov hansel
        if "cos" in STRUCTURE: # J_ij = 1/N (1 + 2.0 kappa cos(theta_ij))
            if verbose:
                print('with cosine structure')
                
            Cij[:, :] = (1.0 + 2.0 * Pij) / Nb

            if SIGMA>0.0:
                if verbose:
                    print('with asymmetry')
                Cij[:, :] =  Cij + SIGMA * numba_normal((Nb, Nb), SEED) / Nb            
        else:
            Cij[:, :] = 1.0 / Nb

    else:
        if verbose:
            print('sparse connectivity')
        
        if 'cos' in STRUCTURE:
                
            if "spec" in STRUCTURE:
                if verbose:
                    print('with spec cosine structure')
                Pij[:, :] = Pij / np.sqrt(Kb)
            else:
                if verbose:
                    print('with strong cosine structure')       
            
            Pij[:, :] = (Kb / Nb) * (1.0 + 2.0 * Pij)
            Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Pij)
        else:
            Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Kb/Nb)
            
    return Cij
