from numpy import ones
from numpy.random import rand


def random_eta(k):
    eta = (rand(k) + ones(k) / k) / 2
    eta = eta / eta.sum()
    return eta

