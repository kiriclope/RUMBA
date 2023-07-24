from numba import jit
import numpy as np
import time


@jit
def strided_method(ar):
    a = np.concatenate((ar, ar[:-1]))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L-1:], (L, L), (-n, n))


@jit(nopython=True, parallel=True)
def connectivity(K, N, structure=None, **kwargs):

    if structure is None:
        Cij = (np.random.rand(N, N) < K / N)
    elif structure == 'ring':
        theta = np.linspace(0, 2*np.pi, N)
        theta_ij = strided_method(theta)
        cos_ij = np.cos(theta_ij)

        Cij = (np.random.rand(N, N) < K / N * (1 + kwargs['sigma'] * cos_ij))

    id_post = []
    n_post = np.zeros(N)

    for j in range(N):  # pre
        for i in range(N):  # post
            if Cij[i, j]:
                id_post.append(i)  # id of post for pre j
                n_post[j] += 1

    id_post = np.array(id_post)

    idx_post = np.zeros(N)
    for i in range(N-1):
        idx_post[i+1] = idx_post[i] + n_post[i]

    return id_post, idx_post, n_post


@jit
def input_update(inputs, post_neurons, spiked, **kwargs):

    inputs[0] *= EXP_DT_TAU[0]
    inputs[1] *= EXP_DT_TAU[1]

    inputs[0, post_neurons[0, spiked[:NE]]] += J[0, 0]
    inputs[1, post_neurons[0, spiked[NE:]]] += J[0, 1]
    inputs[0, post_neurons[1, spiked[:NE]]] += J[1, 0]
    inputs[1, post_neurons[1, spiked[NE:]]] += J[1, 1]

    return inputs


@jit
def volt_update(volt, **kwargs):

    volt[:NE] *= EXP_DT_TAU_MEM[0]
    volt[NE:] *= EXP_DT_TAU_MEM[1]

    volt[:NE] += DT_TAU[0] * (net_inputs[:NE] + VL[0])
    volt[NE:] += DT_TAU[1] * (net_inputs[NE:] + VL[1])

    return volt


@jit
def run():

    id_post, idx_post, n_post = connectivity(K, N)

    for i in range(DURATION):

        volt = volt_update(volt, net_inputs, **params)

        spiked = (volt>VR)

        inputs = input_update(inputs, post_neurons, spiked, **params)

        net_inputs = ff_inputs + inputs[0] + inputs[1]


if __name__ == '__main__':

    start = time.perf_counter()
    connectivity(1, 1, None)
    end = time.perf_counter()

    print("Elapsed (with compilation) = {}s".format((end - start)))

    start = time.perf_counter()
    dict = {"sigma": .25}

    connectivity(2000, 10000, structure='ring', **dict)
    end = time.perf_counter()

    print("Elapsed (with compilation) = {}s".format((end - start)))
