from itertools import permutations
import pdb
import scipy
import numpy
from numpy.linalg import det, inv, norm
from numpy import log, trace, dot, zeros, array, arange, inf, size, pi, e, sqrt, diag, max, vstack, min, sum, meshgrid, linspace, ceil, sort
from numpy.random import permutation
from scipy.misc import derivative
import numpy as np
from itertools import count, izip, islice
from numpy.core.multiarray import count_nonzero
from minimize import fastGradientProjectionStream

import time

# noinspection PyUnresolvedReferences
from optimized_rw_core import projCore, make_positive


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def nop(*a, **kwa):
    pass


def segmentationError(modelStates, S):
    """Return (p,e), where p is a permutation of 1...max(S) minimizing e the
    number of differences between the partition by modelStates and that in S.
    """
    assert (len(modelStates) == S.max())
    Snew = segmentData(modelStates)
    permutationsAndScores = [(p, sum(Snew + 1 !=
                                     array([p[s - 1] for s in S])) / float(S.size))
                             for p in permutations(range(1, S.max() + 1))]
    bestPIndex = array(zip(*permutationsAndScores)[1]).argmin()
    return permutationsAndScores[bestPIndex]


def segmentData(models):
    return np.argmin(array([st.squaredLosses() for st in models]), 0)


def meanMinSquaredLoss(stateSet):
    vecs = [s.squaredLosses() for s in stateSet]
    return bestOfErrorVecs(vecs).mean()


def reportAndBoundStream(stream, maxIters=100, report=nop):
    for (k, xk) in izip(xrange(maxIters), stream):
        if report:
            report(xk, k)
    return xk


def numericalMinRate(f, proj, x0, eps=1e-6, theta=1., maxIter=100, report=nop):
    return projectedSubgradient(lambda x: functionGradient(f, x, eps), proj, x0, maxIter=maxIter, report=report)


def numericalMin(f, projection, x0, maxIters=100, report=nop):
    return fastGradientProjectionMethod(f, lambda x: 0, lambda x: functionGradient(f, x, 1e-9), projection, x0,
                                        maxIters=maxIters, report=report)


def projectedSubgradient(sgf, proj, x0, theta=1., maxIters=100, report=None):
    return reportAndBoundStream(projectedSubgradientStream(sgf, proj, x0, theta=theta), report)

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def subset(L, cardinality):
    k, n = L.shape
    orderedIdx = arange(n)
    messedIdx = permutation(orderedIdx)
    rightIdx = messedIdx[:cardinality]
    rightData = L[:,rightIdx]
    return rightData


def subgradientPolyakCFM(f, sgf, x0, lowerBounder, gamma=1, maxIters=100, report=None):
    ''' Use the Polyak step size and CFM direction update rule. '''
    xk = x0
    gk = sgf(x0)
    sk = gk
    lb = lowerBounder(xk)
    fb = f(xk)
    # Using optimal step size for fixed number of iterations
    for k in arange(maxIters) + 1:
        if report:
            report(xk, k)
        sko = sk
        gko = gk
        gk = sgf(xk)
        #betak = 0.25
        '''betak = (- gamma * gk.dot(sko)/(norm(sko)**2)
                 if sko.dot(gk) < 0 else 0 )'''
        betak = (- gamma * gk.dot(gko) / (norm(gko) ** 2)
                 if gko.dot(gk) < 0 else 0 )
        #sk =  gk + betak*sko
        sk = gk + betak * gko
        nfb = f(xk)
        fb = fb if fb < nfb else nfb
        if nfb < fb:
            nlb = lowerBounder(xk)
            lb = lb if lb > nlb else nlb

        alphak = 0.5 * (fb - lb) / (norm(sk)) #**2 ) #* (k ** -0.66)
        xk = xk - alphak * sk

    return xk


def projectToSimplexNewtonReference(v, target=1.):
    """ The orthogonal projection of v into the simplex is of form sum(nonNegativePart(v-a)) for some a. The function a->sum(nonNegativePart(v-a)) is decreasing and convex.
    Then we can use a newton's iteration, non-smoothness notwithstanding. """
    a = min(v) - target
    f = sum(nonNegativePart(v - a)) - target

    while f > 10 ** -8:
        diff = v - a
        nn = nonNegativePart(diff)
        nnc = count_nonzero(nn)
        nns = nn.sum()
        f = nns - target
        df = nnc #(1.0 * (diff > 0)).sum()
        a = a + f / (df + 1e-6)

    return nonNegativePart(v - a)


def projectToSimplexNewton(v, target=1., into=None):
    """ The orthogonal projection of v into the simplex is of form sum(nonNegativePart(v-a)) for some a. The function a->sum(nonNegativePart(v-a)) is decreasing and convex.
    Then we can use a newton's iteration, non-smoothness notwithstanding. """
    a = v.min() - target
    nnc, nns = projCore(v, a)
    f = nns - target
    #f = sum(nonNegativePart(v - a)) - target
    r = 0

    while f > 10 ** -8:
        r += 1
        if r > 100:
            print ("f: %s, nnc = %s, nns = %s, a = %s" % (f, nnc, nns, a))
            #pdb.set_trace()
        if r > 120:
            raise(Exception("120 iterations for a projection is plenty. f: %s, nnc = %s, nns = %s, a = %.50g" % (f, nnc, nns, a)))
        #print("."),
        nnc, nns = projCore(v, a)
        f = nns - target
        df = nnc #(1.0 * (diff > 0)).sum()
        if df == 0:
            raise(Exception("df should never be 0. f: %s, nnc = %s, nns = %s, a = %s" % (f, nnc, nns, a)))
        if a + (f / df) == a:
            break
        a = a + f / df
    #print(" ")
    if into is None:
        pv = v - a
        make_positive(pv)
        return pv
    else:
        into[:] = v
        into -= a
        make_positive(into)


def projectToSimplex(v):
    lowerBound = min(v) - 1
    upperBound = max(v)
    for i in arange(0, 60):
        mid = (lowerBound + upperBound) / 2
        if sum(nonNegativePart(v - mid)) > 1:  # Decreasing function in mid
            lowerBound = mid
        else:
            upperBound = mid
    return nonNegativePart(v - mid)


def nonNegativePart(v):
    """ A copy of v with negative entries replaced by zero. """
    return (v + abs(v)) / 2


def nonNegativePartII(v):
    """ A copy of v with negative entries replaced by zero. """
    return max(vstack((v, zeros(v.shape))), 0)


def weightedData(data, weights):
    return data * sqrt(weights)


def squaredColNorm(data):
    return (np.power(data, 2)).sum(0)


def bestOfErrorVecs(vecs):
    return array(vecs).min(0)


def PCA(data, dim):
    """Lower data to dimension dim.
    Given data of shape (idim,numPoints), return with shape (dim,numPoints),
    using principal components."""
    U, d, Vt = np.linalg.svd(data.T, full_matrices=False)
    return (U[:, :dim] * d[:dim]).T


def naturals():
    return count(1)


def medianAbsLoss(stateSet):
    vecs = [s.squaredLosses() ** 0.5 for s in stateSet]
    return np.median(bestOfErrorVecs(vecs))


def approximateRootOfIncreasingFunction(g):
    lower = 0
    upper = 1
    mid = 0.5
    for _ in arange(30):
        if g(mid) > 0:
            upper = mid
        elif g(mid) < 0:
            lower = mid
        else:
            return mid
        mid = (upper + lower) / 2
    return lower


def quadratic(vector, psdMatrix):
    return dot(dot(vector, psdMatrix), vector)


def logRatioOf2dGaussians(mu1, Sigma1, mu2, Sigma2, location):
    relloc1 = location - mu1
    relloc2 = location - mu2
    return -0.5 * (log(det(Sigma1) / det(Sigma2)) + quadratic(relloc1, inv(Sigma1)) - quadratic(relloc2, inv(Sigma2)))


def gaussian2dDensity(mu, Sigma, location): #mu,location are vectors of size 2, Sigma a 2x2 covariance matrix
    relloc = location - mu
    return numpy.power(det(Sigma), -0.5) * numpy.power(e, -dot(relloc, inv(Sigma)).dot(relloc) / 2) / 2 / pi


def KLdiv(P, Q):
    (res, err) = scipy.integrate.dblquad(lambda x, y: P(x, y) * log(P(x, y) / Q(x, y)), -inf, inf, lambda x: -inf,
                                         lambda x: inf)
    print err
    return res


def KLtwoGaussians(mu0, Sigma0, mu1, Sigma1):
    mudiff = mu1 - mu0
    S1Inv = inv(Sigma1)
    k = size(Sigma0, 0)
    return 0.5 * (trace(S1Inv.dot(Sigma0)) + mudiff.dot(S1Inv).dot(mudiff) - log(det(Sigma0) / det(Sigma1)) - k)


def KLdivGaussians(mu1, Sigma1, mu2, Sigma2):
    P = lambda x, y: gaussian2dDensity(mu1, Sigma1, array([x, y]))
    logPdivQ = lambda x, y: logRatioOf2dGaussians(mu1, Sigma1, mu2, Sigma2, array([x, y]))
    (res, err) = scipy.integrate.dblquad(lambda x, y: P(x, y) * logPdivQ(x, y), -inf, inf, lambda x: -inf,
                                         lambda x: inf)
    print err
    return res


def ei(i, n):
    ei = zeros(n)
    ei[i] = 1
    return ei


def functionGradient(f, x0, delta0):
    n = x0.shape[0]
    return array([derivative(lambda dx: f(x0 + ei(i, n) * dx), 0, delta0) for i in arange(x0.shape[0])])


def functionHessian(f, x0, delta0):
    return functionGradient(lambda x1: functionGradient(f, x1, delta0), x0, delta0)


def oneSidedBinarySearch(f, x0, initialStep=0.0001):
    '''Assume a pseudo convex function, with a (global) minimum at bounded x>x0, at distance of scale approximately 0.001.'''
    bestVal = f(x0)
    stepSize = initialStep
    nextPos = x0 + stepSize
    nextVal = f(nextPos)
    while nextVal < bestVal:
        bestVal = nextVal
        stepSize = stepSize * 8
        nextPos = x0 + stepSize
        nextVal = f(nextPos)

    return scipy.optimize.fminbound(f, x0, nextPos, xtol=1e-13)


def functionAtArray(xlims, ylims, f, res=100):
    X, Y = meshgrid(linspace(xlims[0], xlims[1], res), linspace(ylims[0], ylims[1], res))
    res = zeros(X.shape)
    l0 = X.shape[0]
    l1 = X.shape[1]
    for i in arange(l0):
        for j in arange(l1):
            res[i, j] = f(X[i, j], Y[i, j])
    return res


def fastGradientProjectionMethod(f, g, gradf, proxg, x0, maxIters=100, report=nop, initLip=None):
    '''The (fast) proximal gradient method requires a gradient of f, and a prox
    operator for g, supplied by gradf and proxg respectively.'''
    for (k, xk) in izip(xrange(maxIters), fastGradientProjectionStream(f, g, gradf, proxg, x0, initLip=initLip)):
        if report:
            report(xk, k)
    return xk


def slowGradientProjectionMethod(f, g, gradf, proxg, x0, maxIters=100, report=nop):
    '''The (fast) proximal gradient method requires a gradient of f, and a prox
    operator for g,f, supplied by sgf and proxg respectively.'''
    Lipk = 1.
    eta = 2.
    xko = x0
    xk = x0
    yk = x0
    tk = 1

    def F(x):
        return f(x) + g(x)

    def Q(Lip, px, x):
        d = (px - x).flatten() # treat all matrices as vectors
        return f(x) + gradf(x).flatten().dot(d) + Lip * (norm(d) ** 2) / 2 + g(px)

    def P(Lip, x):
        return proxg(Lip, x - gradf(x) / Lip)

    '''Non standard extension: expanding line search to find an initial estimate of Lipschitz constant'''
    for _ in range(5):
        pyk = P(Lipk, yk)
        if F(pyk) > Q(Lipk, pyk, yk):
            break
        Lipk = Lipk / (eta ** 4)
    '''Start standard algorithm'''
    for k in range(maxIters):
        if report:
            report(xk, k)

        while True:
            pyk = P(Lipk, yk)
            if F(pyk) <= Q(Lipk, pyk, yk):
                if F(pyk) > F(yk):
                    pdb.set_trace()
                break
            Lipk = Lipk * eta

        xk = pyk
        yk = xk
    return xk
