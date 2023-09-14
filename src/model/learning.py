import numpy as np

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
def numba_update_Cij(Cij, rates, ALPHA=1, ETA_DT=0.01):
    norm = (Cij>0) * ALPHA * Cij * rates**2
    Cij = Cij + ETA_DT * (np.outer(rates, rates) - norm)
    print(np.mean(Cij))

    return Cij

class HebbianLearningModels:

    def __init__(self):
        self.eta = 0
        self.alpha = 0
        
    def hebbian_rule(self, wij, ri, rj):

        Dwij = self.eta * ri * rj

    def oja_rule(self, wij, ri, rj):  # sum wij2 = 1/alpha
        Dwij = self.eta * (ri * rj - self.alpha * ri * rj * wij)

    # def bcm_rule(self, wij, ri, rj):
