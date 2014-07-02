from numpy.random import randn
from numpy import diff, allclose
from testUtilities import random_eta
from minL2PenalizedLossOverSimplex import weightsForMultipleLosses2BlockMinimization, penalizedMultipleWeightedLoss2, penalizedMultipleWeightedLoss2Gradient, gOfTs, gOfTGrad
from utility import functionGradient

k = 3
n = 30
alpha = n * 3
L = abs(randn(k, n))


def testLossGradientAndBlockwise():
    W = weightsForMultipleLosses2BlockMinimization(L, alpha, maxIters=1000)
    loss = penalizedMultipleWeightedLoss2(L, W, alpha)
    Wg = penalizedMultipleWeightedLoss2Gradient(L, W, alpha)

    # Optimal W for a given modelset: for non zero weights all gradients should be equal (otherwise we can improve within a model by moving weight without changing the linear sum constraint)
    print diff(sorted(Wg[W > 0]))
    print sorted(Wg[W > 0])
    print Wg * (W > 0)
    assert (diff(sorted(Wg[W > 0])) > (0.1 / n / k)).sum() == k - 1


def testLossGradient():
    eta = random_eta(k)
    W = weightsForMultipleLosses2BlockMinimization(L, alpha, eta=eta, maxIters=1)
    Wg = penalizedMultipleWeightedLoss2Gradient(L, W, alpha, eta=eta)
    lossAtWeight = lambda ws: penalizedMultipleWeightedLoss2(L, ws.reshape((k, n)), alpha, eta=eta)
    WgAppx = functionGradient(lossAtWeight, W.reshape((n * k)), 10e-9).reshape((k, n))

    # Our gradient calculation should be close to a numerical approximation
    assert allclose(WgAppx, Wg, atol=10e-5)


def testHGradient():
    # at a general position, any subgradient is the gradient
    eta = random_eta(k)
    ts = randn(k)

    def dualValue(t):
        return gOfTs(L, alpha, t, eta)
    dg = gOfTGrad(L, alpha, ts, eta)
    dgAppx = functionGradient(dualValue, ts, 10e-9)

    # could still fail at points of non-differentiability
    assert allclose(dgAppx, dg, 10e-5)

