import numpy as np
from time import perf_counter
from yaml import safe_load

from rate_class import Network

if __name__ == "__main__":

    config = safe_load(open("./config_bump.yml", "r"))

    start = perf_counter()
    name = config['FILE_NAME']

    for gain in [.25, .5, .75, 1.0, 1.25, 1.5, 1.75]:

        config['TF_GAIN'] = gain

        for i_simul in range(20, 250):

            config['FILE_NAME'] = name + "_gain_%.2f_id_%d" % (gain, i_simul)
            model = Network(**config)
            model.run()

    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
