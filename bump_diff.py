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

    config['DPHI'] = 1

    for i_simul in range(0, 1000):

        if config['IF_LOAD_MAT'] == 0:
            config['IF_LOAD_MAT'] = 1
            config['IF_SAVE_MAT'] = 0

        name = 'dist'
        distance = 'close'

        config['Iext'] = [24.0]

        if distance == 'close':
            config['DPHI'] = -0.25
        else:
            config['DPHI'] = -1.0

        config['FILE_NAME'] = name + "_%s_I0_%.2f_id_%d" % (distance, config['Iext'][0], i_simul)
        print('trial', i_simul, config['FILE_NAME'])
        model = Network(**config)
        model.run()

        if distance == 'close':
            config['DPHI'] = 0.25
        else:
            config['DPHI'] = 1.0

        config['FILE_NAME'] = name + "2_%s_I0_%.2f_id_%d" % (distance, config['Iext'][0], i_simul)
        print('trial', i_simul, config['FILE_NAME'])
        model = Network(**config)
        model.run()

        config['Iext'] = [14.0]

        if distance == 'close':
            config['DPHI'] = -0.25
        else:
            config['DPHI'] = -1.0

        config['FILE_NAME'] = name + "_%s_I0_%.2f_id_%d" % (distance, config['Iext'][0], i_simul)
        print('trial', i_simul, config['FILE_NAME'])
        model = Network(**config)
        model.run()

        if distance == 'close':
            config['DPHI'] = 0.25
        else:
            config['DPHI'] = 1.0

        config['FILE_NAME'] = name + "2_%s_I0_%.2f_id_%d" % (distance, config['Iext'][0], i_simul)
        print('trial', i_simul, config['FILE_NAME'])
        model = Network(**config)
        model.run()
