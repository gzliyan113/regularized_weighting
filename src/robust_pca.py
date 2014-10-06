from numpy.ma import floor
from numpy.random.mtrand import randn
from minL2PenalizedLossOverSimplex import weightsForLosses
from numpy import ones, zeros, allclose, zeros_like, diag
from weightedModelTypes import MultiPCAState
from utility import squaredColNorm, nonNegativePart


def weighted_pca(X, k, beta):
    samples, dim = X.shape
    alpha = beta * samples
    w = ones(samples)/samples
    w_old = zeros(samples)

    # actually, always written before read
    s = None
    while not allclose(w, w_old):
        w_old = w
        s = MultiPCAState(X.T, w, modelParameters={'d': k})
        w = weightsForLosses(s.squaredLosses(), alpha)
    return s


def shrinkage_pca(X, k, outlier_cost):
    samples, dim = X.shape
    u = ones(samples) / samples
    O = zeros_like(X)
    O_old = O + 1
    s = MultiPCAState(X.T, u, modelParameters={'d': k})
    while not allclose(O, O_old):
        O_old = O
        R = s.projection.dot(X.T)
        O = soft_shrink_rows(X - R.T, outlier_cost / 2)
        s = MultiPCAState(X.T - O.T, u, modelParameters={'d': k})
    return s


def soft_shrink_rows(X, length):
    """ Reduce the Euclidean norm of rows of X by length (down to zero). """
    curr_row_lengths = squaredColNorm(X.T)
    unit_length_rows = (X.T / curr_row_lengths).T
    desired_lengths = nonNegativePart(curr_row_lengths - length)
    return (unit_length_rows.T * desired_lengths).T


def plain_pca(X, k):
    samples, dim = X.shape
    u = ones(samples) / samples
    return MultiPCAState(X.T, u, modelParameters={'d': k})


def noisy_data(total_samples, dimension, big_dims, noisy_proportion, noise_level):
    scales = ones(dimension) / 10
    scales[:big_dims] = 1
    X = randn(total_samples, dimension).dot(diag(scales))
    noisy_samples = int(floor(total_samples * noisy_proportion))
    X[:noisy_samples, :] += randn(noisy_samples, dimension) * noise_level
    return X