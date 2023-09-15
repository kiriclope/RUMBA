import numpy as np
from pandas import DataFrame, concat

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
        
def nd_numpy_to_nested(X, N_POP=2, IF_STP=0):
def nd_numpy_to_nested(X, N_POP=2, IF_STP=0):
    """
    Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints)
    into pandas DataFrame (with time series as pandas Series in cells)
    Parameters
    ----------
    X : NumPy ndarray, input
    Returns
    -------
    pandas DataFrame
    """
    df_list = []
    
    if IF_STP:
        if N_POP == 2:
            variables = ["rates", "ff", "h_E", "h_I", "u_stp", "x_stp", "A_stp"]
        else:
            variables = ["rates", "ff", "h_E", "u_stp", "x_stp", "A_stp"]
    else:
        if N_POP == 2:
            variables = ["rates", "ff", "h_E", "h_I"]
        else:
            variables = ["rates", "ff", "h_E"]

    idx = np.arange(0, X.shape[1], 1)
    
    for i_time in range(X.shape[0]):
        df_i = pd.DataFrame(X[i_time, :, 1:], columns=variables)
        df_i["neurons"] = idx
        df_i["time"] = X[i_time, 0, 0]
        df_list.append(df_i)

    df = pd.concat(df_list, ignore_index=True)

    return df
    
    # """Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints)
    # into pandas DataFrame (with time series as pandas Series in cells)
    # Parameters
    # ----------
    # X : NumPy ndarray, input
    # Returns
    # -------
    # pandas DataFrame
    # """

    # # print(X.shape)

    # if IF_STP:
    #     if N_POP == 2:
    #         variables = ["rates", "ff", "h_E", "h_I", "u_stp", "x_stp", "A_stp"]
    #     else:
    #         variables = ["rates", "ff", "h_E", "u_stp", "x_stp", "A_stp"]
    # else:
    #     if N_POP == 2:
    #         variables = ["rates", "ff", "h_E", "h_I"]
    #     else:
    #         variables = ["rates", "ff", "h_E"]

    # df = DataFrame()
    # idx = np.arange(0, X.shape[1], 1)
    # for i_time in range(X.shape[0]):
    #     df_i = DataFrame(X[i_time, :, 1:], columns=variables)
    #     df_i["neurons"] = idx
    #     df_i["time"] = X[i_time, 0, 0]

    #     # print(df_i)
    #     df = concat((df, df_i))

    # # print(df)

    # return df
    