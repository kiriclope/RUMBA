import sys
from time import perf_counter

import numpy as np
from rate_numba.model.rate_class import Network
from yaml import safe_load

if __name__ == "__main__":
    name = sys.argv[1]
    config = safe_load(open("./config_" + name + ".yml", "r"))

    start = perf_counter()

    I0_LIST = np.arange(10, 28, 2)

    config["verbose"] = 0
    config["IF_LOAD_MAT"] = 0
    config["IF_SAVE_MAT"] = 1

    config["I1"] = [0]

    name = "diff"

    for i_simul in range(0, 1000):
        for I0 in I0_LIST:
            print("trial", i_simul, "I0", I0)

            if config["IF_LOAD_MAT"] == 0:
                config["IF_LOAD_MAT"] = 1
                config["IF_SAVE_MAT"] = 0

            config["Iext"] = [I0]
            config["FILE_NAME"] = name + "_I0_%.2f_id_%d" % (I0, i_simul)
            model = Network(**config)
            model.run()
    
    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
