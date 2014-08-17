from itertools import izip, count
from numpy import inf, zeros, array, isneginf, hstack, zeros_like
from numpy.linalg import norm
from numpy.core.umath import sqrt

__author__ = 'danielv'


try:
    from cvxoptCuttingPlane import proximalIteration
    def proximalCuttingPlaneStream(f, g, x0, lowerBound=None, stepSize=1.):
        n = x0.shape[0]
        xk = yk = ykn = x0
        bko = f(x0)
        bestLowerBound = -inf if lowerBound is None else lowerBound
        if lowerBound is None:
            A, Z, b = zeros((n, 0)), zeros((n, 0)), zeros(0)
            #A,Z,b = zeros((n,1)),zeros((n,1)),array([-10e12])
        else:
            A, Z, b = zeros((n, 1)), zeros((n, 1)), array([lowerBound])

        while (True):
            newLowerBound = yield yk
            # If a lower bound is added or improved by client, update the constraints.
            if (not (newLowerBound is None)):
                if isneginf(bestLowerBound):
                    A = hstack((zeros((n, 1)), A))
                    Z = hstack((zeros((n, 1)), Z))
                    b = hstack((newLowerBound, b))
                if newLowerBound > bestLowerBound:
                    bestLowerBound = newLowerBound
                b[0] = bestLowerBound
                # Compute a new cutting plane through xk.
            ak = g(xk)
            zk = xk
            bk = f(xk)
            if bk < bko: # Serious step: next turn we'll change the reference point.
                ykn = xk
                bko = bk
                stepSize = stepSize * 2
            else:
                stepSize = stepSize / 2
            ''' Update function approximation (currently grows without bound).'''
            A = hstack((A, ak.reshape(n, 1)))
            Z = hstack((Z, zk.reshape(n, 1)))
            b = hstack((b, array([bk])))

            xk = proximalIteration(yk, 1. / stepSize, A, Z, b)
            #xk = proximalIteration(yk, 1. / (k + 1), A, Z, b)
            yk = ykn
except ImportError:
    pass

def sgdStream(gradf_t, w0, stepsizes):
    w = w0

    for alpha, grad in izip(stepsizes, gradf_t):
        yield w
        w = w - alpha * grad(w)
        

def averageStream(stream):
    aw = stream.next()
    yield aw
    for n,w in enumerate(stream, start=2):
        aw = (w + aw * (n-1)) / float(n)
        yield aw

        
def averageLateWeightingStream(stream):
    aw = stream.next()
    yield aw
    for n, w in enumerate(stream, start=2):
        aw = (2*w + aw * (n-1)) / float(n + 1)
        yield aw


def regularizedDualAveragingStream(gradf_t, prox, w0, L, gamma):
    """ Essentially Algorithm 3 (Accelerated RDA method) from the paper
 Dual Averaging Methods for Regularized Stochastic Learning and Online Optimization by Lin Xiao, JMLR 2010.

- L is the Lipschitz constant for the gradients.
- prox(g,C) is a function that solves the proximal operator argmin_w {<g,w> + Psi(w) + C*h(w)}, where Psi is the regularization function and h is a strongly convex localizer (e.g. squared distance from a minimizer of Psi).
- gradf_t is a sequence of instances of an unbiased estimators for gradients of the base function. For example, each may correspond to a different random minibatch of data points.
- w0 is the minimizer of h. """
    w = w0
    v = w
    A = 0
    gt = zeros_like(w)
    yield w

    for t, gradf in izip(count(1.), gradf_t):
        alpha_t = t / 2
        beta_t = gamma * ((t + 1) ** 1.5) / 2
        # Calculate coefficients
        A += alpha_t
        theta_t = alpha_t / A
        # Compute the query point
        u = (1 - theta_t) * w + theta_t * v
        # Query stochastic gradient, update weighted average gradient
        gt = (1 - theta_t) * gt + theta_t * gradf(u)
        # Solve for the exploration point
        v = prox(gt, (L + beta_t) / A)
        # Interpolate for w
        w = (1 - theta_t) * w + theta_t * v
        yield w


def fastGradientProjectionStream(f, g, gradf, proxg, x0, initLip=None):
    '''The (fast) proximal gradient method requires a gradient of f, and a prox
    operator for g, supplied by gradf and proxg respectively.'''
    if initLip is None:
        Lipk = 1.
    else:
        Lipk = initLip

    eta = 2.
    xko = x0
    xk = x0
    yk = x0
    tk = 1
    yield xk

    def F(x):
        return f(x) + g(x)

    Fxko = F(xko)

    def Q(Lip, px, x):
        d = (px - x).flatten() # treat all matrices as vectors
        return f(x) + gradf(x).flatten().dot(d) + Lip * (norm(d) ** 2) / 2 + g(px)

    def P(Lip, x):
        return proxg(Lip, x - gradf(x) / Lip)

    '''Non standard extension: expanding line search to find an initial estimate of Lipschitz constant'''
    for k in range(5):
        pyk = P(Lipk, yk)
        Fpyk = F(pyk)
        if Fpyk > Q(Lipk, pyk, yk):
            break
        if Fpyk < Fxko:
            yield pyk
        Lipk /= eta ** 4
    '''Start standard algorithm'''
    xk = pyk

    while True:
        while True:
            pyk = P(Lipk, yk)
            Fyk = F(pyk)
            if Fyk <= Q(Lipk, pyk, yk):
                break
            Lipk *= eta

        zk = pyk
        tkn = (1 + sqrt(1 + 4 * (tk ** 2))) / 2
        if Fyk <= Fxko: # Fyk is F(zk)=F(pyk); Fxko is F(xko)
            xk = zk
            Fxk = Fyk
        else:
            xk = xko
            Fxk = Fxko
        yk = xk + (zk - xk) * tk / tkn + (xk - xko) * (tk - 1) / tkn
        Fxko = Fxk
        xko = xk
        tk = tkn
        yield xk

def projectedSubgradientStream(sgf, proj, x0, theta=1.):
    """ Minimize a function f whose subgradient is sgf over
    a convex set of radius theta and orthogonal projection operator proj,
    starting at x0. """
    theta = float(theta)
    xk = x0
    # Using optimal step size for fixed number of iterations
    for k in count(1):
        yield xk
        gk = sgf(xk)
        'Theoretically motivated step size, sometimes decays too slowly'
        tk = sqrt(2 * theta / k) / norm(gk)
        uncon = xk - tk * gk
        xk = proj(uncon)
