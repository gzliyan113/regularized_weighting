from scipy.io import loadmat
from numpy import flatnonzero, cumsum, unique, reshape, argsort, size
from numpy.linalg import svd
import numpy as np
import weightedModelTypes


def hopkinsDatasetOptimalErrors(data, S):
    errors = []
    for i in unique(S):
        subset = (S == i)
        q = np.count_nonzero(subset)
        U, s, Vt = svd(data[:, subset])
        dim = min(flatnonzero(cumsum(s) / sum(s) > 0.99)) + 1
        totalError = sum(weightedModelTypes.MultiPCAState(data[:, subset],
                                                          ones(q) / q, dim).squaredLosses())
        errors.append(totalError)
    return errors


def hopkinsDatasetDimensions(data, S):
    dims = []
    for i in unique(S):
        subset = (S == i)
        U, s, Vt = svd(data[:, subset])
        dim = min(flatnonzero(cumsum(s) / sum(s) > 0.99)) + 1
        dims.append(dim)
    return dims


def loadSet(hopkinsLocation, name):
    location = hopkinsLocation + '/' + name + '/'
    fileName = name + '_truth.mat'
    try:
        allVars = loadmat(location + fileName)
        S = allVars['s'] # group per point.
        Yraw = allVars['y'] # homogenous screen coordinates per point per frame
        (dummy, P, F) = Yraw.shape # P = #points; F=#frames
        newOrder = reshape(argsort(S, axis=0), size(S))
        Ysorted = Yraw[:, newOrder, :] # ensure points are in increasing order of group
        Ydehom = Ysorted[:2, :, :] # remove the scale=1 column
        Yreshaped = reshape(Ydehom.swapaxes(0, 1), (P, 2 * F)) # (2,P,F) => (P,2F)
        Y = Yreshaped.byteswap().newbyteorder().T
        return (Y, reshape(S[newOrder], size(S)))
    except IOError:
        return (array([]), array([]))
