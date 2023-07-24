from time import perf_counter

import numpy as np
from rate_numba.model.rate_class import Network
from yaml import safe_load

if __name__ == "__main__":
    config = safe_load(open("./config_bump.yml", "r"))

    start = perf_counter()
    name = config["FILE_NAME"]

    for Iext in np.linspace(1, 20, 20):
        config["Iext"] = [Iext]

        for var_ff in np.linspace(0, 100, 10):
            config["VAR_FF"] = [var_ff]

            for id in range(10):
                config["FILE_NAME"] = name + "_I0_%.2f_S0_%.2f_id_%d" % (
                    Iext,
                    var_ff,
                    id,
                )
                model = Network(**config)
                model.run()

    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
