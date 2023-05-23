import sys
import numpy as np
from time import perf_counter
from yaml import safe_load

from rate_class import Network

if __name__ == "__main__":

    name = sys.argv[1]
    config = safe_load(open("./config_" + name + ".yml", "r"))

    start = perf_counter()

    for i_simul in range(0, 250):
        config['I0'] = [16.0]
        config['FILE_NAME'] = name + "_first_I0_16.00_id_%d" % (i_simul)
        model = Network(**config)
        model.run()

        config['I0'] = [26.0]
        config['FILE_NAME'] = name + "_first_I0_26.00_id_%d" % (i_simul)
        model = Network(**config)
        model.run()

    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
