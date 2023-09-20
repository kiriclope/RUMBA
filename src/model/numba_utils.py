import numpy as np
from numba import njit
from math import erf

@njit(parallel=True, fastmath=True, cache=True)
def numba_erf(x):
    """Numba erf."""    
    res = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        res[i] = erf(x[i])
    return res


@njit(parallel=True, fastmath=True, cache=True)
def numba_TF(x, thresh=15):
    """numba_TF."""
    # if tfname=='NL':
    # return x * (x > thresh)
    # elif tfname=='Sig':
    return thresh * (0.5 * (1.0 + numba_erf(x / np.sqrt(2.0)))).astype(np.float64)


@njit(parallel=True, fastmath=True, cache=True)
def numba_update_ff_inputs(
    ff_inputs, ff_inputs_0, EXP_DT_TAU_FF, DT_TAU_FF, VAR_FF, FF_DYN=0
):
    """Update ff inputs."""
    if FF_DYN == 1:
        ff_inputs = ff_inputs * EXP_DT_TAU_FF[0]
        ff_inputs = ff_inputs + DT_TAU_FF[0] * ff_inputs_0
    elif FF_DYN == 2:
        ff_inputs[:] = (
            np.sqrt(VAR_FF[0]) * np.random.normal(0, 1.0, ff_inputs.shape[0])
            + ff_inputs_0
        )
    else:
        ff_inputs = ff_inputs_0

    return ff_inputs


@njit( parallel=True, fastmath=True, cache=True)
def numba_update_inputs(Cij, rates, inputs, csumNa, EXP_DT_TAU_SYN, SYN_DYN=1):
    """Update recurrent inputs."""
    if SYN_DYN == 0:
        for i_pop in range(inputs.shape[0]):
            inputs[i_pop] = np.dot(
                Cij[:, csumNa[i_pop] : csumNa[i_pop + 1]],
                rates[csumNa[i_pop] : csumNa[i_pop + 1]],
            )
    else:
        for i_pop in range(inputs.shape[0]):
            inputs[i_pop] = inputs[i_pop] * EXP_DT_TAU_SYN[i_pop]
            inputs[i_pop] = inputs[i_pop] + np.dot(
                Cij[:, csumNa[i_pop] : csumNa[i_pop + 1]],
                rates[csumNa[i_pop] : csumNa[i_pop + 1]],
            )

    return inputs


@njit(parallel=True, fastmath=True, cache=True)
def numba_update_rates(
    rates,
    ff_inputs,
    inputs,
    inputs_NMDA,
    thresh,
    csumNa,
    EXP_DT_TAU_MEM,
    DT_TAU_MEM,
    RATE_DYN=0,
    IF_NMDA=0,
):
    net_inputs = ff_inputs

    for i_pop in range(inputs.shape[0]):
        net_inputs = net_inputs + inputs[i_pop]

    if IF_NMDA:
        for i_pop in range(inputs.shape[0]):
            net_inputs = net_inputs + inputs_NMDA[i_pop]
    
    if RATE_DYN == 0:
        rates = numba_TF(net_inputs, thresh)
    else:
        for i_pop in range(inputs.shape[0]):
            rates[csumNa[i_pop] : csumNa[i_pop + 1]] = (
                rates[csumNa[i_pop] : csumNa[i_pop + 1]] * EXP_DT_TAU_MEM[i_pop]
            )
            rates[csumNa[i_pop] : csumNa[i_pop + 1]] = rates[
                csumNa[i_pop] : csumNa[i_pop + 1]
            ] + DT_TAU_MEM[i_pop] * numba_TF(
                net_inputs[csumNa[i_pop] : csumNa[i_pop + 1]],
                thresh[csumNa[i_pop] : csumNa[i_pop + 1]],
            )

    return rates
