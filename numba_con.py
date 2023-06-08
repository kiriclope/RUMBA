import numpy as np
from numba import jit, njit
from scipy.special import i0
from scipy.ndimage import convolve

@jit(nopython=False, parallel=False, fastmath=True, cache=True)
def numba_sample_proba(p, size):

    shape = size
    if np.count_nonzero(p > 0) < size:
        raise ValueError("Fewer non-zero entries in p than size")

    n_uniq = 0
    p = p.copy()
    found = np.zeros(shape).astype(np.int64)
    flat_found = found.ravel()

    while n_uniq < size:
        x = np.random.rand(size - n_uniq)
        if n_uniq > 0:
            p[flat_found[0:n_uniq]] = 0
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        new = cdf.searchsorted(x, side='right')
        _, unique_indices = np.unique(new, return_index=True)
        unique_indices.sort()
        new = new.take(unique_indices)
        flat_found[n_uniq:n_uniq + new.size] = new
        n_uniq = n_uniq + new.size

    idx = found
    # print(idx)
    return idx

# @jit(nopython=False, parallel=False, fastmath=True, cache=True)
# def numba_random_choice(K, N, p, DUM=1):
#     Cij = np.zeros((N, N), dtype=np.float64)

#     if DUM==0:
#         id_pattern = np.random.choice(N, np.int64(K/2.0), replace=False)
#     else:
#         id_pattern = numba_sample_proba(p[0], np.int64(K/2.0))

#     for i in range(N):
#         if i>0:
#             id_pattern = (id_pattern + 1) % N

#         for j in id_pattern:
#             Cij[i, j] = 1.0
#             Cij[j, i] = 1.0

#     return Cij

@jit(nopython=False, parallel=False, fastmath=True, cache=True)
def numba_random_choice(K, N, p, DUM=1):

    Cij = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        if DUM==0:
            id_pres = np.random.choice(N, np.int64(K), replace=False)
        else:
            id_pres = numba_sample_proba(p[i], np.int64(K))

        for j in id_pres:
            Cij[i, j] = 1.0

    return Cij


@jit(nopython=False, parallel=False, fastmath=True, cache=True)
def numba_reciprocal_fixed(K, N, p, Pij):

    Cij = np.zeros((N, N), dtype=np.float64)
    idx = np.arange(N)

    if p>0:
        id_recip = np.random.choice(idx, np.int64(p*K), replace=False)
        id_not_recip = np.random.choice(idx[~id_recip], np.int64((1.0-p)*K), replace=False)
    else:
        id_recip = []
        id_not_recip = np.random.choice(idx, np.int64(K), replace=False)

    for i in range(N):

        if i>0:
            id_recip = np.roll(id_recip, 1)
            id_not_recip = np.roll(id_not_recip, 1)

        for j in id_recip:
            Cij[i, j] = 1.0
            Cij[j, i] = 1.0

        for j in id_not_recip:
            Cij[i, j] = 1.0

    return Cij*Pij

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def numba_reciprocal(K, N, p, Pij):
    """""
    Based on Rao's code
    """""
    Cij = np.zeros((N, N), dtype=np.float64)

    for j in range(N):
        for i in range(j):
            if np.random.uniform(0.0, 1.0) <= (p * Pij[i, j] + (1.0 - p) * Pij[i, j]**2):
                Cij[i, j] = 1.0
                Cij[j, i] = 1.0
            else:
                if np.random.uniform(0.0, 1.0) <= 2.0 * (1.0 - p) * Pij[i, j] * (1.0 - Pij[i, j]):
                    if np.random.uniform(0.0, 1.0) > 0.5:
                        Cij[i, j] = 1.0
                    else:
                        Cij[j, i] = 1.0

    return Cij


# @jit(nopython=False, parallel=True, fastmath=True, cache=True)
def moving_average(x, w=3) :
    return convolve(x, np.ones(w), mode='reflect') / w


@jit(nopython=False, parallel=True, fastmath=True, cache=True)
def numba_update_local_field(DJij, smooth, EXP_DT_TAU, KAPPA, DT_TAU, ALPHA):

    # DJij = DJij * EXP_DT_TAU
    # DJij = DJij + KAPPA_DT_TAU * (np.outer(smooth, smooth) - norm)

    # DJij = DJij * EXP_DT_TAU
    # DJij = DJij + KAPPA * DT_TAU * np.outer(smooth, smooth)

    # norm = np.sqrt(smooth)
    # DJij = DJij + KAPPA * DT_TAU * np.outer(smooth, smooth)
    # DJij = DJij + KAPPA * DT_TAU * np.outer(smooth, smooth)

    # norm = ALPHA * DJij * smooth**2
    # DJij = KAPPA * DT_TAU * ( np.outer(smooth, smooth) - norm)

    norm = ALPHA * DJij * smooth**2
    DJij = DJij + DT_TAU * ( KAPPA * np.outer(smooth, smooth) - norm)

    return DJij

@jit(nopython=False, parallel=True, fastmath=True, cache=True)
def numba_update_Jij(DJij, cos_mat, EXP_DT_TAU):

    DJij = DJij * EXP_DT_TAU
    DJij = DJij + cos_mat

    return DJij

@jit(nopython=False, parallel=True, fastmath=True, cache=True)
def numba_normal(size, SEED=0):
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


@jit(nopython=False, parallel=True, fastmath=True, cache=True)
def numba_multiple_maps(K, N, KAPPA, N_MAPS):

    KAPPA_ = KAPPA / np.sqrt(K) / N_MAPS

    # Jij = np.zeros((N, N))
    # theta = np.linspace(0.0, 2.0 * np.pi, np.int(K))
    # Jij[:K,:K] = KAPPA_ * np.cos(theta_mat(theta, theta))

    theta = np.linspace(0.0, 2.0 * np.pi, N)
    Jij = KAPPA_ * np.cos(theta_mat(theta, theta))

    for _ in range(N_MAPS-1):

        # theta = np.random.permutation(theta)
        theta = np.linspace(0.0, 2.0 * np.pi, N)
        neurons = np.zeros(np.int(N-K)).astype(np.int64)

        for i in range(N-K):
            neurons[i] = np.random.randint(N)
        # print(neurons)

        theta_ = theta[neurons]
        theta[neurons] = np.random.permutation(theta_)

        DJij = np.cos(theta_mat(theta, theta))
        Jij = Jij + KAPPA_ * DJij

    return Jij

@jit(nopython=False, parallel=True, fastmath=True, cache=True)
def numba_update_DJij(DJij, rates, EXP_DT_TAU, KAPPA, DT_TAU, ALPHA):

    # KAPPA_DT_TAU = KAPPA * DT_TAU

    # norm = ALPHA * DJij * rates**2
    # norm = ALPHA * DJij * rates
    # DJij = DJij + DT_TAU * ( KAPPA * np.outer(rates, rates) - norm)

    # DJij = DJij * EXP_DT_TAU
    # DJij = DJij + DT_TAU * KAPPA * np.outer(rates, rates)

    # DJij = DJij * EXP_DT_TAU
    # DJij = DJij + DT_TAU * ( KAPPA * np.outer(rates, rates) )

    # DJij = DJij * EXP_DT_TAU
    # DJij = DJij + KAPPA_DT_TAU * np.outer(rates, rates)

    DJij = DJij * EXP_DT_TAU
    norm = np.sum(rates**2)
    DJij = DJij + KAPPA * DT_TAU * np.outer(rates, rates) / norm

    # theta = np.linspace(0.0, 2.0 * np.pi, DJij.shape[0])
    # DJij = DJij * EXP_DT_TAU
    # DJij = DJij + KAPPA_DT_TAU * np.cos(theta_mat(theta, theta))

    # norm = np.sum(rates**2)
    # DJij = KAPPA_DT_TAU * np.outer(rates, rates) / norm

    # print(np.mean(DJij))
    # print(np.mean(DJij), np.mean(KAPPA_DT_TAU * np.outer(rates, rates)))

    return DJij

@jit(nopython=False, parallel=True, fastmath=True, cache=True)
def gaussian(theta, sigma):
    sigma = sigma * np.pi / 180.0

    if sigma > 0:
        return np.exp(-theta**2 / 2.0 / sigma**2) / np.sqrt(2.0 * np.pi) / sigma
    else:
        return np.ones(theta.shape)


# @jit(nopython=False, parallel=True, fastmath=True, cache=True)
# def von_mises(theta, kappa):
#     return np.exp(kappa * np.cos(theta)) / (2.0 * np.pi * i0(kappa))


@jit(nopython=False, parallel=True, fastmath=True, cache=True)
def numba_update_Cij(Cij, rates, ALPHA=1, ETA_DT=0.01):
    norm = (Cij>0) * ALPHA * Cij * rates**2
    Cij = Cij + ETA_DT * (np.outer(rates, rates) - norm)
    print(np.mean(Cij))

    return Cij


@jit(nopython=False, parallel=True, fastmath=True, cache=True)
def theta_mat(theta, phi):
    theta_mat = np.zeros((phi.shape[0], theta.shape[0]))

    twopi = np.float64(2.0 * np.pi)

    for i in range(phi.shape[0]):
        for j in range(theta.shape[0]):
            # theta_mat[i, j] = phi[i] - theta[j]
            dtheta = np.abs(phi[i] - theta[j])
            theta_mat[i, j] = np.minimum(dtheta, twopi - dtheta)

    return theta_mat


@jit(nopython=False, parallel=False, fastmath=True, cache=True)
def strided_method(ar):
    a = np.concatenate((ar, ar[1:]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L-1:], (L, L), (-n, n))


# @jit(nopython=False, parallel=True, fastmath=True, cache=True)
# def numba_i0(*args):
#     return i0(*args)

@jit(nopython=False, parallel=True, fastmath=True, cache=True)
def generate_Cab(Kb, Na, Nb, STRUCTURE='None', SIGMA=1.0, KAPPA=0.5, SEED=0, PHASE=0, verbose=0):

    # np.random.seed(SEED)

    Pij = np.zeros((Na, Nb), dtype=np.float64)
    Cij = np.zeros((Na, Nb), dtype=np.float64)

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
            
        # if 'lateral' in STRUCTURE:
        #     cos2_ij = np.cos(2.0 * theta_ij - PHASE) 
        #     print('lateral')
        #     Pij[:, :] = cos_ij + cos2_ij
        # else:
        #     Pij[:, :] = cos_ij

    if "cos" in STRUCTURE:
        if verbose:
            print('with strong cosine structure')
        Pij[:, :] = Pij * np.float64(KAPPA)

    elif "spec_cos" in STRUCTURE:
        if verbose:
            print('with spec cosine structure')
        Pij[:, :] = Pij * KAPPA / np.sqrt(Kb)
    
    elif "gauss" in STRUCTURE:
        Pij[:, :] = gaussian(theta_ij, np.float64(SIGMA))

    if "all" in STRUCTURE:
        if verbose:
            print('with all to all cosine structure')
        # itskov hansel
        if "cos" in STRUCTURE: # 1/N (1 + cos)
            Cij[:, :] = (1.0 + 2.0 * Pij * KAPPA) / Nb

            if SIGMA>0.0:
                Cij[:, :] =  Cij + SIGMA * numba_normal((Nb, Nb), SEED) / Nb

            # if SIGMA>0.0:
            #     Cij[:, :] =  Cij + np.sqrt(SIGMA) * Pij * numba_normal((Nb,Nb), SEED) / Nb
            #     Cij[:, :] = np.triu(Cij) + np.triu(Cij, 1).T

        elif "id" in STRUCTURE:
            Cij[:, :] = np.identity(Nb)
            
        else:
            Cij[:, :] = 1.0 / Nb 

        # Cij[:, :] = Cij[:, :] * (Cij>=0)

        # Z = Nb / np.sum(Cij, axis=1) / (2.0 * np.pi)
        # Cij[:, :] = Cij[:, :] * Z

        # Cij[:, :] = gaussian(theta_ij, np.float64(SIGMA))
        # Z = Nb / np.sum(Cij, axis=1)
        # Cij[:, :] = Cij[:, :] * Z

    elif STRUCTURE == "gauss":
        if verbose:
            print('with strong gauss proba')
        Z = Kb / np.sum(Pij, axis=1)
        Pij[:, :] = Pij[:, :] * Z
        Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Pij)

    elif "spec_gauss" in STRUCTURE:
        if verbose:
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
        if verbose:
            print("with random spec")
        Cij[:, :] = numba_random_choice(int(Kb-np.sqrt(Kb)), Nb, Pij) + KAPPA * (np.random.rand(Na, Nb) < 2.0 * np.sqrt(Kb) / Nb)
        # Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Kb / Nb) + KAPPA * (np.random.rand(Na, Nb) < 2.0 * np.sqrt(Kb) / Nb)
        # Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < (Kb - KAPPA * np.sqrt(Kb)) /Nb) + 1.0 * (np.random.rand(Na, Nb) < KAPPA * np.sqrt(Kb) /Nb)

    elif "spec_cos_weak" in STRUCTURE:
        if verbose:
            print("with weak cos spec")
        Pij[:, :] = np.sqrt(Kb) * Pij[:, :] + 1.0
        Cij[:, :] = numba_random_choice(int(Kb-np.sqrt(Kb)), int(Nb), Pij)
        Cij[:, :] = Cij[:, :] + SIGMA * (np.random.rand(Na, Nb) < (np.sqrt(Kb) / Nb) * Pij )
        # Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Kb / Nb) + SIGMA * (np.random.rand(Na, Nb) < (2.0 * np.sqrt(Kb) / Nb) * Pij )

    elif "spec_cos_nq" in STRUCTURE:
        if verbose:
            print("with weak cos spec")
        Pij[:, :] =  (Kb / Nb) * (Pij[:, :] + 1.0)
        Cij[:, :] = numba_random_choice(int(Kb), int(Nb), Pij)

    elif "no_quench" in STRUCTURE:
        Cij[:, :] = numba_random_choice(int(Kb), int(Nb), Pij, DUM=0)

    elif "reciprocal" in STRUCTURE:
        if verbose:
            print('with reciprocal connections')
        # Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) <= (1.0 - SIGMA) * Kb / Nb * (1.0 - Kb / Nb))
        # dum = 1.0 * (np.random.rand(Na, Nb) <= (SIGMA * (Kb / Nb) + (1.0 - SIGMA) * Kb**2 / Nb**2))
        # Cij[:, :] = Cij + dum + dum.T

        if "spec" in STRUCTURE:
            Pij[:, :] = Kb / Nb * (Pij + 1.0)
            Cij[:, :] = numba_reciprocal(Kb, Nb, SIGMA, Pij)
        else:
            Cij[:, :] = numba_reciprocal(Kb, Nb, SIGMA,  Kb / Nb * np.ones((Nb, Nb)))

        if "cos" in STRUCTURE:
            Pij[:, :] = (Pij + 1.0)
            Cij[:, :] = Cij * Pij

    elif "recip_nq" in STRUCTURE:
        if verbose:
            print('with fixed reciprocal connections')
        Pij[:, :] = Kb / Nb * (Pij + 1.0)
        Cij[:, :] = numba_reciprocal_fixed(Kb, Nb, SIGMA, Pij)
    else:
        Pij[:, :] = (Kb / Nb) * (2.0 * Pij + 1.0)
        Cij[:, :] = 1.0 * (np.random.rand(Na, Nb) < Pij)

    return Cij
