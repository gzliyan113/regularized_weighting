from numpy import array, allclose, ones
from numpy.random import rand
from minL2PenalizedLossOverSimplex import weightsForMultipleLosses2BlockMinimization, penalizedMultipleWeightedLoss2, weightsForMultipleLosses2GradientProjection, dual1solve5, dual1value, lambda2W, ts2WMixedColumns, tsFromLambda, validateWeights, weightsForMultipleLosses2FISTA, ts2W
from cvxoptJointWeightsOptimization import optimalWeightsMultipleModels2
from testUtilities import random_eta


def testBlockwiseOptimizationVSCvxAverageFormulationVSGradientProjection():
    k, n = (4, 60)
    L = rand(k, n)
    eta = (rand(k) + ones(k)/k)/2
    eta = eta / eta.sum()
    alpha = 100.
    WOTS = optimalWeightsMultipleModels2(L, alpha, eta=eta)
    WBlock = weightsForMultipleLosses2BlockMinimization(L, alpha, eta=eta, maxIters=1000)
    WFISTA = weightsForMultipleLosses2FISTA(L, alpha, eta=eta, maxIters=1000)
    Ws = [WOTS, WBlock, WFISTA]
    [validateWeights(W) for W in Ws]
    losses = array([penalizedMultipleWeightedLoss2(L, W, alpha, eta) for W in Ws])
    print losses
    assert losses.max() - losses.min() < 0.00001
    assert allclose(WOTS, WBlock, atol=0.03)
    assert allclose(WOTS, WFISTA, atol=0.03)
    assert allclose(WFISTA, WBlock, atol=0.03)


def testDualThenLinearBounds():
    "Solve dual then convert to primal. The interval between dual and primal values should be close to the value of an OTS solution."
    k, n = (4, 200)
    L = rand(k, n)
    alpha = 3 * n
    eta = random_eta(k)
    l2 = dual1solve5(L, alpha, eta=eta, maxIters=1000)
    dualLowerBound = dual1value(L, alpha, eta, l2)
    WOTS = optimalWeightsMultipleModels2(L, alpha, eta=eta)
    primalUpperBoundPrecise = penalizedMultipleWeightedLoss2(L, WOTS, alpha, eta=eta)
    print dualLowerBound, primalUpperBoundPrecise

    assert allclose(dualLowerBound, primalUpperBoundPrecise, atol=10e-5)

    WPrecConv = lambda2W(L, l2, alpha, eta)
    ts = tsFromLambda(L, l2)
    WPartitionConv = ts2W(L, ts, alpha, eta)
    primalUpperBoundFull = penalizedMultipleWeightedLoss2(L, WPrecConv, alpha, eta=eta)
    primalUpperBoundQuick = penalizedMultipleWeightedLoss2(L, WPartitionConv, alpha, eta=eta)
    print "duality gap from full : %f" % (primalUpperBoundFull - dualLowerBound)
    print "duality gap from quick: %f" % (primalUpperBoundQuick - dualLowerBound)
    assert allclose(dualLowerBound, primalUpperBoundFull, atol=10e-5)
    assert allclose(dualLowerBound, primalUpperBoundQuick, atol=10e-3)
    assert allclose(WPrecConv, WPartitionConv, atol=10e-2)
