from cvxopt.coneprog import coneqp
from cvxopt.solvers import options
from cvxopt import matrix, spdiag
from numpy import hstack, ones, array


def proximalIteration(y, gamma, A, Z, b):
    """ Columns of A give gradients of linear functions having the values in b
    at the locations of the columns of Z. The maximum of these functions is F,
    and G is the squared distance from y scaled by gamma.
    Then return the minimum of F+G. """
    options['show_progress'] = False
    n, r = A.shape
    P = 2 * gamma * spdiag(list(ones(n)) + [0])
    q = hstack((-2. * gamma * y.T, array([1.])))
    G = hstack((A.T, -1.*ones((r, 1))))
    h = array([ai.dot(zi) - bi for bi, ai, zi in zip(b, A.T, Z.T)])
    resdict = coneqp(P, matrix(q), G=matrix(G), h=matrix(h))
    return array(resdict['x'])[:-1].reshape(n)
