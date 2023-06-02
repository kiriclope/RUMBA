import numpy as np
from sklearn.decomposition import PCA

# generate some sample data
n_trials = 10
n_neurons = 5
n_time = 1000
data = np.random.randn(n_trials, n_neurons, n_time)

# perform PCA and extract the principal components (PCs)
pca = PCA(n_components=n_neurons)
pca.fit(data.reshape(n_trials*n_neurons, n_time))
pcs = pca.transform(data.reshape(n_trials*n_neurons, n_time))

# calculate the fraction of variance explained by each PC
var_exp = pca.explained_variance_ratio_

# calculate the PEV for each neuron and time point
pev = np.zeros((n_neurons, n_time))
for i in range(n_neurons):
    for j in range(n_time):
        pev[i, j] = (var_exp[i] / np.sum(var_exp)) * (np.sum(pcs[:, i]**2) / (n_trials*n_neurons))
