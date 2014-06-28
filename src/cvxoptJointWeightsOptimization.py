from cvxopt.coneprog import coneqp
from cvxopt.solvers import lp
from cvxopt import matrix, sparse, spmatrix, spdiag
from numpy import eye, hstack, zeros, ones, array, vstack, arange, tile


def rowJofK(k, q, j):
    t = zeros((k, q))
    t[j, :] = 1
    return t


def sparseRowJofK(k, q, j):
    return spmatrix(ones(q), q * [j], range(q), (k, q))


def optimalWeightsMultipleModels2raw(losses, alpha, eta=None, initWeights=None, hintDict=None):
    k, n = losses.shape
    eta = eta if eta is not None else ones(k)/k
    alpha = float(alpha)
    if initWeights is not None:
        initvals = {'x': matrix(initWeights.flatten()), 's': matrix(initWeights.flatten())}
    elif hintDict:
        initvals = hintDict
    else:
        initvals = None
    I = spdiag(list(ones(n)))
    B = sparse([I*eta[j] for j in range(k)]).T
    P = 2 * alpha * (B.T * B)
    q = losses.flatten() * tile(eta, (n, 1)).T.flatten() - (alpha * ones(n * k) / (k * n))
    A = sparse([[sparseRowJofK(k, n, j)] for j in range(k)])
    b = ones(k)
    G = -spdiag(list(ones(k * n)))
    h = zeros(k * n)
    return coneqp(P, matrix(q), G=G, h=matrix(h), A=A, b=matrix(b), initvals=initvals)


def optimalWeightsMultipleModels2(losses, alpha, eta=None, initWeights=None, hintDict=None):
    k, q = losses.shape
    res_dict = optimalWeightsMultipleModels2raw(losses, alpha, eta=eta, initWeights=initWeights, hintDict=hintDict)
    return array(res_dict['x']).reshape((k, q))


def optimalWeightsMultipleModelsFixedProfile(losses, averageWeights, eta, rowSums=None, initWeights=None):
    "Min. W'L s.t. each row w_j of W is a distribution with sum rowSums[j], and the mean of all rows is averageWeights."
    k, n = losses.shape
    rowSums = ones(k) if rowSums is None else rowSums
    if initWeights is not None:
        initvals = {'x': matrix(initWeights.flatten()), 's': matrix(initWeights.flatten())}
    else:
        initvals = None
    q = losses.flatten() * tile(eta, (n, 1)).T.flatten()
    Aparts = [[sparseRowJofK(k, n, j), spdiag(list(ones(n) * eta[j]))] for j in range(k)]
    A = sparse(Aparts)
    b = hstack((rowSums, averageWeights))
    G = spdiag(list(-ones(k * n)))
    h = zeros(k * n)
    "Min. q'x s.t. Gx <= h and Ax=b"
    resDict = lp(matrix(q), G=G, h=matrix(h), A=A, b=matrix(b), solver='glpk')
    return array(resDict['x']).reshape((k, n))


def optimalWeightsMultipleModelsFixedProfileNonSparse(losses, averageWeights, initWeights=None):
    k, q = losses.shape
    if initWeights is not None:
        initvals = {'x': matrix(initWeights.flatten()), 's': matrix(initWeights.flatten())}
    else:
        initvals = None
    t = losses.flatten()
    A1 = hstack([rowJofK(k, q, j) for j in range(k)])
    A2 = hstack([eye(q) / k for j in range(k)])
    A = vstack((A1, A2))
    b = hstack((ones(k), averageWeights))
    G = -eye(k * q)
    h = zeros(k * q)
    resDict = lp(matrix(t), G=matrix(G), h=matrix(h), A=matrix(A), b=matrix(b), solver='glpk')
    return array(resDict['x']).reshape((k, q))


def dual1prox(Lip, L, oldLambda):
    k, q = L.shape
    P = Lip * spdiag(list(hstack((zeros(k), ones(q)))))
    Q = matrix(-hstack((ones(k), oldLambda)))
    rows = arange(k * q)
    leftG = [spmatrix([1.] * q * k, range(q * k), [r % k for r in range(q * k)], (k * q, k))]
    rightG = [spmatrix([- 1. / k] * (k * q), range(q * k), [r / k for r in range(k * q)], (k * q, q))]
    G = sparse([leftG, rightG])
    resDict = coneqp(P, Q, G=G, h=matrix(L.T.flatten(), tc='d'))
    return array(resDict['x'])[k:].flatten()


def dual1prox2(Lip, L, oldLambda):
    k, q = L.shape
    P = Lip * spdiag(list(hstack((zeros(k), ones(q)))))
    Q = matrix(-hstack((ones(k), zeros(q))))
    rows = arange(k * q)
    leftG = [spmatrix([1.] * q * k, range(q * k), [r % k for r in range(q * k)], (k * q, k))]
    rightG = [spmatrix([- 1. / k] * (k * q), range(q * k), [r / k for r in range(k * q)], (k * q, q))]
    G = sparse([leftG, rightG])
    resDict = coneqp(P, Q, G=G, h=matrix((L + (oldLambda / k)).T.flatten(), tc='d'))
    return array(resDict['x'])[k:].flatten() + oldLambda
