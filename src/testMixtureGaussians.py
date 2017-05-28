from clusteringBenchmark import make_clustered_points
from simpleInterface import mixtureGaussians
from numpy import sqrt, hstack
from numpy.random import randn
import pdb

d = 1
k = 2
n = 200

#%X = hstack((randn(1,100) * 2,5+randn(1,100) * 1))
X = hstack((randn(1,100) * 2,5+randn(1,100) * 20))
alpha = 3 * (X.max()-X.min()) * n

means, variances, foundLabels = mixtureGaussians(X.T, k, alpha, n_init=1)
