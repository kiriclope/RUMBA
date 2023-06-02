import sys
import numpy as np
from time import perf_counter
from yaml import safe_load

from rate_class import Network

if __name__ == "__main__":

    name = sys.argv[1]
    config = safe_load(open("./config_" + name + ".yml", "r"))

    start = perf_counter()
    # I0_LIST = [4, 6, 8, 32, 34, 36]
    I0_LIST = np.arange(10, 28, 2)
    for i_simul in range(0, 250):
        for I0 in I0_LIST:
            config['Iext'] = [I0]
            config['FILE_NAME'] = name + "3_I0_%.2f_id_%d" % (I0, i_simul)
            model = Network(**config)
            model.run()

    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
