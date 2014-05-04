from numpy import arange, array, sqrt, diag, ones, tile, dot, hstack, zeros, eye, log, pi, diag
from numpy.linalg import lstsq, norm
from numpy.random import randn
import random
import numpy as np
import utility
import pdb

def basisForDataBasedSubspace(data, subspaceDimension):
    U, _, _ = np.linalg.svd(data, full_matrices=False)
    return U[:, :subspaceDimension]


def basisForRandomSubspace(ambientDimension, subspaceDimension):
    # an orthonormal basis, at that
    return basisForDataBasedSubspace(randn(ambientDimension, ambientDimension),
                                     subspaceDimension)


class ScalarGaussianState:
    @staticmethod
    def randomModelLosses(data, k, modelParameters=None):
        # Start with a random point from dataset, with scalar covariance set so that n/k data points are reasonable.
        d, n = data.shape
        means = data[:, random.sample(range(n), k)].T
        L = zeros((k,n))
        for j,m in enumerate(means):
            differences = (data.T - m).T
            squaredDistances = array(utility.squaredColNorm(differences))
            variance = np.percentile(squaredDistances, 100. / k)
            L[j,:] = (d*log(pi*2*variance) + squaredDistances/variance)/2.
        return L

    def __init__(self, theData, theWeights, modelParameters=None):
        self.data = theData
        self.weights = theWeights
        self.d, self.n = self.data.shape

        # find the least squares solution and some useful measures thereof.
        wData = self.data.dot(diag(self.weights))
        self.mean = wData.sum(axis=1)
        differences = (self.data.T - self.mean).T
        squaredDistances = array(utility.squaredColNorm(differences))
        self.variance = squaredDistances.dot(self.weights)
        self.l = (self.d*log(pi*2*self.variance) + squaredDistances/self.variance)/2.

    def nextState(self, newWeights):
        return ScalarGaussianState(self.data, newWeights)

    def squaredLosses(self):
        return self.l

    def badness(self):
        return diag(self.weights).dot(self.squaredLosses())



class MultiPCAState:
    @staticmethod
    def randomModelLosses(data, k, modelParameters):
        d = modelParameters['d']
        n, numPoints = data.shape
        L = array([utility.squaredColNorm(basisForRandomSubspace(n, n - d).T.dot(data))
                   for i in arange(k)])
        return L

    def __init__(self, theData, theWeights, modelParameters):
        self.d = modelParameters['d']
        self.data = theData
        self.weights = theWeights
        self.n, self.numPoints = self.data.shape

        # find the least squares solution and some useful measures thereof.
        wData = utility.weightedData(self.data, self.weights)
        self.U, self.s, V = np.linalg.svd(wData, full_matrices=False)
        # basis for the subspace
        self.basis = self.U[:, :self.d]
        # projection to the subspace
        projt = diag(hstack((ones(self.d), zeros(self.n - self.d))))
        self.projection = dot(dot(self.U, projt), self.U.transpose())

    def nextState(self, newWeights):
        return MultiPCAState(self.data, newWeights, {'d': self.d})

    def squaredLosses(self):
        difference = self.data - dot(self.projection, self.data)
        return utility.squaredColNorm(difference)

    def badness(self):
        return diag(self.weights).dot(self.squaredLosses())

    def effect(self):
        return diag(self.weights).dot(utility.squaredColNorm(self.data))

    def dDimensionalityRatio(self):
        return self.s[self.d - 1] / self.s[self.d]


class MultiLinearRegressionState:
    @classmethod
    def randomModelLosses(cls, data, k, modelParameters):
        regularizationStrength = modelParameters['regularizationStrength']
        (X, Y) = data
        n, numPoints = X.shape
        return array([cls.givenModelLosses(X, Y, randn(n)) for _ in arange(k)])

    @classmethod
    def givenModelLosses(cls, X, Y, r):
        return (r.dot(X) - Y) ** 2

    def __init__(self, data, theWeights, modelParameters):
        self.regularizationStrength = modelParameters['regularizationStrength']
        theDataX, theDataY = data
        self.dataX = theDataX
        self.dataY = theDataY
        self.weights = theWeights
        self.n, self.numPoints = self.dataX.shape
        wDataX = theDataX * tile(theWeights, (self.n, 1))
        wDataY = theDataY * theWeights
        dataM = wDataX.dot(wDataX.T) + self.regularizationStrength * eye(self.n)
        # find the least squares solution and some useful measures thereof.
        self.r, _, _, _ = lstsq(dataM, wDataX.dot(wDataY))

    def nextState(self, newWeights):
        return MultiLinearRegressionState((self.dataX, self.dataY), newWeights,
                                          {'regularizationStrength': self.regularizationStrength})

        # distances from solution

    def squaredLosses(self):
        return self.givenModelLosses(self.dataX, self.dataY, self.r)

    def badness(self):
        return diag(self.weights).dot(self.squaredLosses())

    def effect(self):
        return diag(self.weights).dot(utility.squaredColNorm(self.dataX))


def squaredLossOfLinearFunctional(X, Y, r):
    return (self.r.dot(X) - Y) ** 2


def quadraticKernelMapping(X, weights=None):
    ''' Kernel mapping: x -> (xx^T,x,1).
    Assumes the iterator over x gives a data point each time. '''
    weights = weights if weights is not None else ones(len(X))
    return array([np.hstack((np.outer(x, x).flatten() * (w ** 0.5),
                             x.flatten() * (w ** 0.5),
                             array([1]) * (w ** 0.5)))
                  for (x, w) in zip(X, weights)])


def applyQuadratic(X, c):
    return quadraticKernelMapping(X).dot(c)


class ClusteringState:
    @staticmethod
    def randomModelLosses(data, k, modelParameters=None):
        d, n = data.shape
        centers = data[:, random.sample(range(n), k)]
        L = zeros((k, n))
        for j in range(k):
            L[j, :] = utility.squaredColNorm(data - centers[:,j].reshape((d, 1)))
        return L

    def __init__(self, data, theWeights, parameters=None):
        self.data = data
        self.n, self.q = data.shape
        self.weights = theWeights
        self.center = self.data.dot(self.weights.T)

    def nextState(self, newWeights):
        return ClusteringState(self.data, newWeights)

        # distances from solution

    def squaredLosses(self):
        return utility.squaredColNorm(self.data - self.center.reshape((self.n, 1)))

    def badness(self):
        return diag(self.weights).dot(self.squaredLosses())


class MultiQuadraticRegressionState:
    @staticmethod
    def randomModelLosses(data, k, modelParameters=None):
        dataX, dataY = data

        n, numPoints = dataX.shape
        u = ones(numPoints) / numPoints

        randModelCoeffs = [randn(n ** 2 + n + 1) for _ in arange(k)]
        L = array([((dataY - quadraticKernelMapping(dataX.T, u).dot(c)) ** 2).flatten()
                   for c in randModelCoeffs])
        return L


    def __init__(self, data, theWeights, parameters=None):
        "No parameters are expected or used"
        (theDataX, theDataY) = data
        self.dataX = theDataX
        self.dataY = theDataY
        self.weights = theWeights
        self.n, self.numPoints = self.dataX.shape

        # find the least squares solution and some useful measures thereof.

        # Kernel mapping: x -> (xx^T,x,1)
        self.z = quadraticKernelMapping(self.dataX.T, self.weights)

        (self.r, _, _, _) = lstsq(self.z, self.dataY * sqrt(self.weights))

    def nextState(self, newWeights):
        return MultiQuadraticRegressionState((self.dataX, self.dataY), newWeights)

        # distances from solution

    def squaredLosses(self):
        allYs = quadraticKernelMapping(self.dataX.T).dot(self.r)
        allDiffs = allYs - self.dataY
        squaredDiffs = allDiffs ** 2
        return squaredDiffs.flatten()

    def badness(self):
        return diag(self.weights).dot(self.squaredLosses())
