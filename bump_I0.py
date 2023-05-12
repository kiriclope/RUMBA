import numpy as np
from time import perf_counter
from yaml import safe_load

from rate_class import Network

if __name__ == "__main__":

    config = safe_load(open("./config_bump.yml", "r"))

    start = perf_counter()
    name = config['FILE_NAME']

    for I0 in np.arange(10, 32, 2):

        config['Iext'] = [I0]

        for i_simul in range(100, 250):

            config['FILE_NAME'] = name + "_I0_%.2f_id_%d" % (I0, i_simul)
            model = Network(**config)
            model.run()

    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
