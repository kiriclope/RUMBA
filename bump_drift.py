import sys
import numpy as np
from time import perf_counter
from yaml import safe_load

from rate_class import Network

if __name__ == "__main__":

    name = sys.argv[1]
    config = safe_load(open("./config_" + name + ".yml", "r"))

    start = perf_counter()

    list_cue = [45, 90, 135, 180, 225, 270, 315]
    # list_cue = [0]

    config['verbose'] = 0
    config['SIGMA'] = 2.0
    config['IF_LOAD_MAT'] = 0
    config['IF_SAVE_MAT'] = 1

    config['DURATION'] = 5.0
    config['I1'] = [0]

    name = 'inhom'

    for i_simul in range(10, 50):
        for i_cue in list_cue:

            config['PHI0'] = i_cue
            config['FILE_NAME'] = name + "_sig_%.3f_cue_%d_id_%d" % (config['SIGMA'], i_cue, i_simul)

            print('trial', i_simul, 'cue', i_cue, config['FILE_NAME'])

            model = Network(**config)
            model.run()

            if config['IF_LOAD_MAT'] == 0:
                config['IF_LOAD_MAT'] = 1
                config['IF_SAVE_MAT'] = 0

    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
