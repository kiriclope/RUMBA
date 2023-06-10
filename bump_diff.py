import sys
import numpy as np
from time import perf_counter
from yaml import safe_load

from rate_class import Network

if __name__ == "__main__":

    name = sys.argv[1]
    config = safe_load(open("./config_" + name + ".yml", "r"))

    start = perf_counter()

    config['verbose'] = 0
    config['IF_LOAD_MAT'] = 0
    config['IF_SAVE_MAT'] = 1

    config['DPHI'] = .25

    for i_simul in range(0, 100):
        print('trial', i_simul)

        if config['IF_LOAD_MAT'] == 0:
            config['IF_LOAD_MAT'] = 1
            config['IF_SAVE_MAT'] = 0

        config['Iext'] = [14.0]
        config['FILE_NAME'] = name + "_close_I0_%.2f_id_%d" % (config['Iext'][0], i_simul)
        model = Network(**config)
        model.run()

        config['Iext'] = [26.0]
        config['FILE_NAME'] = name + "_close_I0_%.2f_id_%d" % (config['Iext'][0], i_simul)
        model = Network(**config)
        model.run()
