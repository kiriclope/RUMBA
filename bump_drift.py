import sys
import numpy as np
from time import perf_counter
from yaml import safe_load

from rate_class import Network

if __name__ == "__main__":

    name = sys.argv[1]
    config = safe_load(open("./config_" + name + ".yml", "r"))

    start = perf_counter()

    list_cues = [45, 90, 135, 180, 225, 270, 315]

    for i_cue in list_cues:
        config['PHI0'] = i_cue
        for i_simul in range(0, 50):
            config['FILE_NAME'] = name + "3_cue_%d_id_%d" % (i_cue, i_simul)
            model = Network(**config)
            model.run()


    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
