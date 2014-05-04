from numpy import array, allclose
from numpy.random import rand
from minL2PenalizedLossOverSimplex import weightsForMultipleLosses2BlockMinimization, penalizedMultipleWeightedLoss2, weightsForMultipleLosses2GradientProjection, dual1solve5, dual1value, lambda2W, ts2WMixedColumns, tsFromLambda
from cvxoptJointWeightsOptimization import optimalWeightsMultipleModels2


def testBlockwiseOptimizationVSCvxAverageFormulationVSGradientProjection():
    k, n = (4, 60)
    L = rand(k, n)
    alpha = 10.
    WOTS = optimalWeightsMultipleModels2(L, alpha)
    WBlock = weightsForMultipleLosses2BlockMinimization(L, alpha, maxIters=1000)
    WGradProj = weightsForMultipleLosses2GradientProjection(L, alpha, maxIters=1000)

    losses = array([penalizedMultipleWeightedLoss2(L, WOTS, alpha) for W in [WOTS, WGradProj, WBlock]])
    assert losses.max() - losses.min() < 0.00001
    assert allclose(WOTS, WBlock, atol=0.001)
    assert allclose(WOTS, WGradProj, atol=0.001)
    assert allclose(WGradProj, WBlock, atol=0.001)


def testDualThenLinearBounds():
    "Solve dual then convert to primal. The interval between dual and primal values should be close to the value of an OTS solution."
    k, n = (4, 60)
    L = rand(k, n)
    alpha = 10.
    l2 = dual1solve5(L, alpha, maxIters=100)
    dualLowerBound = dual1value(L, alpha, l2)
    W0 = lambda2W(L, l2, alpha)
    ts = tsFromLambda(L, l2)
    W1 = ts2WMixedColumns(L, ts, alpha)
    primalUpperBoundFull = penalizedMultipleWeightedLoss2(L, W0, alpha)
    primalUpperBoundQuick = penalizedMultipleWeightedLoss2(L, W1, alpha)
    print "duality gap from full : %f" % (primalUpperBoundFull - dualLowerBound)
    print "duality gap from quick: %f" % (primalUpperBoundQuick - dualLowerBound)
    assert allclose(dualLowerBound, primalUpperBoundFull, atol=10e-5)
    assert allclose(dualLowerBound, primalUpperBoundQuick, atol=10e-5)
    assert allclose(W0, W1, atol=10e-4)
