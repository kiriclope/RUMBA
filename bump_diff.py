import numpy as np
from time import perf_counter
from yaml import safe_load

from rate_class import Network

if __name__ == "__main__":

    config = safe_load(open("./config_bump.yml", "r"))

    start = perf_counter()
    name = config['FILE_NAME']

    for i_simul in range(0, 250):
        config['FILE_NAME'] = name + "_%d" % (i_simul)
        model = Network(**config)
        model.run()

    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
