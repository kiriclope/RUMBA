import numpy as np
from numba import jit
from scipy.ndimage import convolve

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

    Pij = np.zeros((Na, Nb), dtype=np.float64)
    Cij = np.zeros((Na, Nb), dtype=np.float64)
    Jij = np.ones((Na, Nb), dtype=np.float64)

    if verbose:
        print('random connectivity')

    if STRUCTURE != 'None':
        theta = np.linspace(0.0, 2.0 * np.pi, Nb)
        phi = np.linspace(0.0, 2.0 * np.pi, Na)
        
        # if 'perm' in STRUCTURE:
        #     print('permuted map')
        #     theta = np.random.permutation(theta) - np.pi
        #     # phi = np.random.permutation(phi)
        
        theta = theta.astype(np.float64)
        phi = phi.astype(np.float64)

        theta_ij = theta_mat(theta, phi)
        if 'lateral' in STRUCTURE:
            if verbose:
                print('lateral')
            cos_ij = np.cos(theta_ij - np.pi)
        else:
            cos_ij = np.cos(theta_ij - PHASE)

        if 'cos' in STRUCTURE:
            Pij[:, :] = cos_ij

    if "ring" in STRUCTURE:
        if verbose:
            print('with strong cosine structure')
        Pij[:, :] = Pij * np.float64(KAPPA)

    elif "spec_cos" in STRUCTURE:
        if verbose:
            print('with spec cosine structure')
        Pij[:, :] = Pij * KAPPA / np.sqrt(Kb)


    if "all" in STRUCTURE:
        if verbose:
            print('with all to all cosine structure')
        # itskov hansel
        if "cos" in STRUCTURE: # 1/N (1 + cos)
            Cij[:, :] = (1.0 + 2.0 * Pij * KAPPA) / Nb

            if SIGMA>0.0:
                Cij[:, :] =  Cij + SIGMA * numba_normal((Nb, Nb), SEED) / Nb

        elif "id" in STRUCTURE:
            Cij[:, :] = np.identity(Nb)
            
        else:
            Cij[:, :] = 1.0 / Nb 

    else:
        Pij[:, :] = (Kb / Nb) * (2.0 * Pij + 1.0)
        Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Pij)

    return Cij
