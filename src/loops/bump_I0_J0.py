from time import perf_counter

import numpy as np
from rate_numba.model.rate_class import Network
from yaml import safe_load

if __name__ == "__main__":
    config = safe_load(open("./config_itskov.yml", "r"))

    start = perf_counter()
    name = config["FILE_NAME"]

    for I0 in np.arange(0, 22, 2):
        config["Iext"][0] = I0 + 5.0

        for J0 in np.arange(0, 22, 2):
            config["Jab"][0] = J0 / 11.0

            config["FILE_NAME"] = name + "_I0_%.2f_J0_%.2f" % (I0, J0)
            model = Network(**config)
            model.run()

    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
