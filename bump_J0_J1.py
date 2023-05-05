import numpy as np
from time import perf_counter
from yaml import safe_load

from rate_class import Network

if __name__ == "__main__":

    config = safe_load(open("./config_bump.yml", "r"))

    start = perf_counter()
    name = config['FILE_NAME']

    for Jab in range(1, 11):

        config['Jab'] = [-Jab]

        for kappa in np.linspace(0, 1, 11):

            config['KAPPA'] = kappa

            config['FILE_NAME'] = name + "_Jab_%.2f_kappa_%.2f" % (Jab, kappa)
            model = Network(**config)
            model.run()


    end = perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
