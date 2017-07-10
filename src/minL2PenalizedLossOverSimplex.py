# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 09:56:38 2012

@author: - Daniel Vainsencher
"""
from numpy import (ones, vstack, zeros, arange, allclose, sum, array,
                   ones_like, nan_to_num, sort, log, ceil, mean, sqrt,
                   diag, median, zeros_like, abs, tile, cumsum, floor, log2, infty, nonzero, maximum, ndarray, where)
from numpy.linalg import norm
from numpy.random import randint, randn, permutation
from scipy.optimize import fminbound
import pdb
from time import time
from minimize import proximalCuttingPlaneStream, fastGradientProjectionStream, sgdStream
from itertools import repeat, count
from utility import projectToSimplexNewton, projectToSimplexNewtonReference, nonNegativePart, ei, oneSidedBinarySearch, projectedSubgradient, subgradientPolyakCFM, nop, reportAndBoundStream, fastGradientProjectionMethod, slowGradientProjectionMethod, subset
from cvxoptJointWeightsOptimization import optimalWeightsMultipleModelsFixedProfile, dual1prox2
from lambdaAndWinnersCache import LambdaAndWinnersTree

# noinspection PyUnresolvedReferences
from optimized_rw_core import lambdaAndWinners


def weightsForLosses(losses, alpha, goal=None):
    """ Given losses vector and regularization strength alpha, find the optimal weights. Regularization is squared
    euclidean from goal, a uniform vector by default. This solves the special case k=1 of assigning weight
    semi-analytically. """
    q = losses.shape[0]
    h = goal if goal is not None else ones(q) / q
    unconstrainedSolution = h - losses / (2 * alpha)
    return projectToSimplexNewton(unconstrainedSolution)


def adjustedWeightsForWeightedLosses(L, W, eta, alpha, j):
    k, n = L.shape
    u = ones(n) / n
    all = arange(k)
    abj = all[all != j] # all but j
    goal = (eta[j] ** -1.) * (u - (W[abj, :].T * eta[abj]).T.sum(0))
    return weightsForLosses(L[j, :], alpha * eta[j], goal=goal)


def weightsForMultipleLosses2(L, alpha, maxIters=100, report=nop):
    return weightsForMultipleLosses2DualPrimal(L, alpha, report=report)


def weightsCombinedForAM(L, alpha, eta, ts0=None, W0=None, maxIters=None, report=nop):
    return weightsCombined(L, alpha, eta, 10, lowDimSolveStream, ts2W,
                           weightsForMultipleLosses2FISTAStream, ts0=ts0, W0=W0,
                           max_iters=10, relative_duality_gap=1e-2,
                           report=report)


def weightsCombinedConsistent(L, alpha, eta, ts0=None, maxIters=1000, report=nop):
    return weightsCombined(L, alpha, eta, 10, lowDimSolveStream, ts2W,
                           weightsForMultipleLosses2FISTAStream, ts0=ts0,
                           max_iters=maxIters, relative_duality_gap=1e-9,
                           report=report)


def weightsCombinedConsistent2(L, alpha, eta, ts0=None, maxIters=1000, report=nop):
    return weightsCombined(L, alpha, eta, 10, lowDimSolveStream, ts2W3,
                           weightsForMultipleLosses2FISTAStream, ts0=ts0,
                           max_iters=maxIters, relative_duality_gap=1e-9,
                           report=report)


def weightsCombined(L, alpha, eta, ratio, ts_stream_maker, to_primal,
                    primal_stream_maker, ts0=None, W0=None, max_iters=None,
                    relative_duality_gap=1e-2, report=nop):
    """
     Allocate half of time to improving our best primal solution, and half
     of time to generating dual solutions and converting them to primal
     candidates.
    """
    k, n = L.shape
    ts = ts0 if ts0 is not None else randn(k)
    W = W0 if W0 is not None else ones((k, n)) / n
    primStr = primal_stream_maker(L, alpha, eta, W)
    W = primStr.next()
    primal_value = penalizedMultipleWeightedLoss2(L, W, alpha, eta=eta)
    report('p0', W, None, None)

    # Enough iterations for FISTA to converge from the default W.
    max_iters = max_iters if max_iters is not None else (
        int(ceil(k * sqrt(alpha / n / (primal_value * relative_duality_gap)))))

    primalTime = .0
    dualTime = .0

    tsStr = ts_stream_maker(L, alpha, eta, ts0=ts0)

    report('d0', None, None, ts)
    dualTime, ts = timedNext(dualTime, tsStr)
    report('d1', None, None, ts)
    dual_value = gOfTs(L, alpha, ts, eta)
    old_dual_value = dual_value

    doPrimal = True
    primal_iter = 0
    dual_iter = 0
    while primal_iter < max_iters:
        if doPrimal:
            primal_iter += 1
            primalTime, W = timedNext(primalTime, primStr)
            primal_value = penalizedMultipleWeightedLoss2(L, W, alpha, eta=eta)
            report('p1' + str(primal_iter), W, None, None)
            if dualTime < primalTime:
                doPrimal = False
        else:
            dual_iter += 1
            old_dual_value = dual_value
            old_ts = ts
            for _ in range(ratio):
                dualTime, ts = timedNext(dualTime, tsStr, -primal_value)
            dual_value = gOfTs(L, alpha, ts, eta)
            if dual_value < old_dual_value:
                print("old dual: %f new dual: %f (got worse)" %(old_dual_value, dual_value))
                #pdb.set_trace()

            report('dc' + str(dual_iter), None, None, ts)
            if dualTime > primalTime:
                doPrimal = True
                # Create a primal candidate from dual, replace the current primal if better.
                Wc = to_primal(L, ts, alpha, eta)
                primal_c_value = penalizedMultipleWeightedLoss2(L, Wc, alpha, eta=eta)
                if primal_value > primal_c_value:
                    #print("Switching primal, gain %f" % (primal_value - primal_c_value))
                    W = Wc
                    primal_value = primal_c_value
                    primStr = primal_stream_maker(L, alpha, eta, W)
                    report('pc' + str(primal_iter), W, None, None)

        gap = primal_value - dual_value
        if gap < primal_value * relative_duality_gap:
            break
    '''
    if primal_iter < max_iters:
        print("Stopping: primal: %f, dual: %f, max_rel_gap: %f, gap: %f, iteration= %i out of potential: %i" %
              (primal_value, dual_value, relative_duality_gap, gap, primal_iter, max_iters))
    else:
        print("Ran out of %i iterations. Primal: %f, dual: %f, max_rel_gap: %f, gap: %f" %
              (max_iters, primal_value, dual_value, relative_duality_gap, gap))
              '''
    return W, ts


def weightsDualPrimal(L, alpha, eta, ratio, tsStreamMaker, toPrimal, ts0=None, maxIters=1000, dualityGapGoal=1e-6,
                      report=nop):
    k, q = L.shape
    lowDimDualStr = tsStreamMaker(L, alpha, eta, ts0=ts0)
    ts = lowDimDualStr.next()
    for i in xrange(maxIters):
        W = toPrimal(L, ts, alpha, eta)
        primalValue = penalizedMultipleWeightedLoss2(L, W, alpha, eta=eta)
        report('p1' + str(i), W, None, None)
        for _ in range(ratio):
            ts = lowDimDualStr.send(-primalValue)
        dualValue = gOfTs(L, alpha, ts, eta)
        report('dc' + str(i), None, None, ts)
        if primalValue - dualValue < dualityGapGoal:
            break
    return W


def dualPrimalSlowAndSafe(L, alpha, eta, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    def tsToW(L, ts, alpha, eta):
        return lambda2W(L, lambda2FromTs(L, ts), alpha, eta)

    return weightsDualPrimal(L, alpha, eta, 50, lowDimSolveStream, tsToW, ts0=ts0,
                             maxIters=maxIters, dualityGapGoal=dualityGapGoal,
                             report=report)


def dualPrimalSlowAndScalable(L, alpha, eta, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    def tsToW(L, ts, alpha, eta):
        return ts2WFista(L, ts, alpha, eta, )

    return weightsDualPrimal(L, alpha, eta, 10, lowDimSolveStream, ts2WFista, ts0=ts0,
                             maxIters=maxIters, dualityGapGoal=dualityGapGoal,
                             report=report)


def dualPrimalFastAndExperimental(L, alpha, eta, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    return weightsDualPrimal(L, alpha, eta, 10, lowDimSolveStream, ts2W2, ts0=ts0,
                             maxIters=maxIters, dualityGapGoal=dualityGapGoal,
                             report=report)


def dualPrimalFastAndImprecise(L, alpha, eta, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    return weightsDualPrimal(L, alpha, eta, 10, lowDimSolveStream, ts2W, ts0=ts0,
                             maxIters=maxIters, dualityGapGoal=dualityGapGoal,
                             report=report)


def weightsForMultipleLosses2DualPrimal(L, alpha, eta, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    k, q = L.shape
    lowDimDualStr = lowDimSolveStream(L, alpha, eta, ts0=ts0)
    ts = lowDimDualStr.next()
    for i in xrange(maxIters):
        W = ts2WMixedColumns(L, ts, alpha, eta)
        primalValue = penalizedMultipleWeightedLoss2(L, W, alpha)
        report('p1' + str(i), W, None, None)
        ts = lowDimDualStr.send(-primalValue)
        dualValue = gOfTs(L, alpha, eta, ts)
        report('dc' + str(i), None, None, ts)
        if primalValue - dualValue < dualityGapGoal:
            break
    return W


def weightsForMultipleLosses2DualPrimal2(L, alpha, eta, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    k, q = L.shape
    lowDimDualStr = lowDimSolveStream(L, alpha, eta, ts0=ts0)
    ts = lowDimDualStr.next()
    report('dc2', None, None, ts)
    for i in xrange(maxIters):
        W = ts2W(L, ts, alpha, eta)
        primalValue = penalizedMultipleWeightedLoss2(L, W, alpha, eta)
        report('p1' + str(i), W, None, None)
        for h in range(10):
            ts = lowDimDualStr.send(-primalValue)
            report('dc2i' + str(i) + 'h' + str(h), None, None, ts)
        dualValue = gOfTs(L, alpha, ts, eta)
        report('dc2i' + str(i), None, None, ts)
        if primalValue - dualValue < dualityGapGoal:
            break
    return W


def weightsForMultipleLosses2DualPrimal3(L, alpha, eta, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    k, q = L.shape
    lowDimDualStr = dualPrimalCoordinateWise(L, alpha, eta, ts0=ts0)
    ts = lowDimDualStr.next()
    report('dc-', None, None, ts.copy())
    for i in xrange(maxIters):
        W = ts2W(L, ts, alpha, eta)
        primalValue = penalizedMultipleWeightedLoss2(L, W, alpha, eta)
        report('p1' + str(i), W, None, None)
        ts = lowDimDualStr.send(-primalValue)
        dualValue = gOfTs(L, alpha, ts, eta)
        report('dc' + str(i), None, None, ts)
        if primalValue - dualValue < dualityGapGoal:
            break
    return W


def weightsForMultipleLosses2DualPrimal3c(L, alpha, eta, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    k, q = L.shape
    lowDimDualStr = dualCoordinateWiseStream(L, alpha, eta, ts0=ts0)
    ts = lowDimDualStr.next()
    report('dc-', None, None, ts.copy())
    for i in xrange(maxIters):
        W = ts2W(L, ts, alpha, eta)
        primalValue = penalizedMultipleWeightedLoss2(L, W, alpha, eta)
        report('p1' + str(i), W, None, None)
        ts = lowDimDualStr.send(-primalValue)
        dualValue = gOfTs(L, alpha, ts, eta)
        report('dc' + str(i), None, None, ts)
        if primalValue - dualValue < dualityGapGoal:
            break
    return W


def weightsForMultipleLosses2DualPrimal4(L, alpha, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    k, q = L.shape
    lowDimDualStr = lowDimSolveStream(L, alpha, ts0=ts0)
    ts = lowDimDualStr.next()
    report('dp', None, None, ts)
    for i in xrange(maxIters):
        W = ts2WMixedColumnsCorrectedSupport(L, ts, alpha)
        primalValue = penalizedMultipleWeightedLoss2(L, W, alpha)
        report('p1' + str(i), W, None, None)
        for h in range(10):
            ts = lowDimDualStr.send(-primalValue)
            report('dp4' + str(i) + 'h' + str(h), None, None, ts)
        dualValue = gOfTs(L, alpha, ts)
        report('dp4' + str(i), None, None, ts)
        if primalValue - dualValue < dualityGapGoal:
            break
    return W


def weightsForMultipleLosses2DualPrimal5(L, alpha, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    k, q = L.shape
    lowDimDualStr = lowDimCoordinateMinimizationStream(L, alpha, ts0=ts0)
    ts = lowDimDualStr.next()
    for i in xrange(maxIters):
        W = ts2WMixedColumnsCorrectedSupport(L, ts, alpha)
        primalValue = penalizedMultipleWeightedLoss2(L, W, alpha)
        report('p1' + str(i), W, None, None)
        ts = lowDimDualStr.send(-primalValue)
        dualValue = gOfTs(L, alpha, ts)
        report('dc' + str(i), None, None, ts)
        if primalValue - dualValue < dualityGapGoal:
            break
    return W


def weightsForMultipleLosses2DualPrimal6(L, alpha, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    k, q = L.shape
    lowDimDualStr = lowDimCoordinateMinimizationStream2(L, alpha, ts0=ts0)
    ts = lowDimDualStr.next()
    for i in xrange(maxIters):
        W = ts2WMixedColumnsCorrectedSupport(L, ts, alpha)
        primalValue = penalizedMultipleWeightedLoss2(L, W, alpha)
        report('p1' + str(i), W, None, None)
        ts = lowDimDualStr.send(-primalValue)
        dualValue = gOfTs(L, alpha, ts)
        report('dc' + str(i), None, None, ts)
        if primalValue - dualValue < dualityGapGoal:
            break
    return W


def weightsForMultipleLosses2OldPrimalDual(L, alpha, l2=None, W=None, maxPrimalIters=900, maxIters=None,
                                           dualityGapGoal=1e-7, report=nop):
    '''Combine calls to primal and dual optimizers to get certified solution.'''
    k, q = L.shape
    primalItersPerRound = 30
    W = ones((k, q)) / q if (W is None) else W
    l2 = zeros(q) if (l2 is None) else l2
    #print('Starting weight finding optimization for k=%d q=%d alpha=%f gap goal=%g' % (k,q,alpha,dualityGapGoal))
    maxIters = maxIters if not maxIters is None else maxPrimalIters / primalItersPerRound
    for i in xrange(maxIters):
        report(i, W, l2, None)
        primal = penalizedMultipleWeightedLoss2(L, W, alpha)
        dual = dual1value(L, alpha, ones(k) / k, l2)
        #print('Primal - dual = gap; %g - %g = %g' % (primal,dual,primal - dual))
        if primal - dual > dualityGapGoal:
            l2 = dual1solve5(L, alpha, fixedLowerBound=-primal, lambda2=l2, maxIters=10)
            report('dual ' + str(i), W, l2, None)
            W = weightsForMultipleLosses2FISTA(L, alpha, W=W,
                                               maxIters=primalItersPerRound)
        else:
            primal = penalizedMultipleWeightedLoss2(L, W, alpha)
            dual = dual1value(L, alpha, ones(k) / k, l2)
            #print('Achieved primal - dual = gap: %g - %g = %g' % (primal,dual,primal - dual))
            return W
            #print('Ran out at primal - dual = gap: %g - %g = %g' % (primal,dual,primal - dual))
    return W


def timedNext(timeSoFar, stream, value=None):
    startTime = time()
    item = stream.send(value)
    return (time() - startTime + timeSoFar, item)


def timedLambda(timeSoFar, lam):
    startTime = time()
    item = lam()
    return (time() - startTime + timeSoFar, item)


def weightsForMultipleLosses2TimedStreams(L, alpha, eta=None, l2=None, W=None, maxPrimalIters=900, maxIters=None,
                                          dualityGapGoal=1e-6, report=nop):
    '''Combine calls to primal and dual optimizers to get certified solution.'''
    k, q = L.shape
    eta = eta if eta is not None else ones(k) / k
    W = ones((k, q)) / q if (W is None) else W
    primalPointStream = weightsForMultipleLosses2FISTAStream(L, alpha, eta=eta, W0=W)
    primalTime = .0
    dualTime = .0

    #print('Starting weight finding optimization for k=%d q=%d alpha=%f gap goal=%g' % (k,q,alpha,dualityGapGoal))
    maxIters = maxIters if not maxIters is None else 2000
    primal = penalizedMultipleWeightedLoss2(L, W, alpha, eta=eta)

    #dualPointStream = lowDimSolveStream(L, alpha, lambda2=l2, fixedLowerBound=-primal)
    dualPointStream = dual1solve5Stream(L, alpha, eta=eta, lambda2=l2, fixedLowerBound=-primal)
    l2init = zeros(q) if (l2 is None) else l2
    l2 = dualPointStream.next()
    dual = dual1value(L, alpha, eta, l2)
    for i in xrange(maxIters):
        report(i, W, l2, None)
        #print('Primal - dual = gap; %g - %g = %g' % (primal,dual,primal - dual))
        if primal - dual > dualityGapGoal:
            if dualTime < primalTime:
                #dualLam = lambda : dual1solve5(L, alpha,fixedLowerBound=-primal, lambda2=l2, maxIters=10)
                dualTime, l2 = timedNext(dualTime, dualPointStream, -primal)
                dual = dual1value(L, alpha, eta, l2)
                report('dual ' + str(i), W, l2, None)
                #print("Dual time: %f val: %f" % (dualTime, dual))
            else:
                primalTime, W = timedNext(primalTime, primalPointStream)
                primal = penalizedMultipleWeightedLoss2(L, W, alpha, eta=eta)
                #if dualTime < primalTime:
                #print("Primal time: %f val: %f" % (primalTime, primal))
        else:
            print('Achieved primal - dual = gap: %.6f - %.6f = %.6f' % (primal, dual, primal - dual))
            return W, l2
    print('Ran out at primal - dual = gap: %g - %g = %g' % (primal, dual, primal - dual))
    return W, l2


def primalDual(L, alpha, W=None, l2=None, dualityGapGoal=1e-7, maxIters=100, report=None):
    k, q = L.shape
    W = ones((k, q)) / q if (W is None) else W
    primal = penalizedMultipleWeightedLoss2(L, W, alpha)
    l2 = zeros(q) if (l2 is None) else l2
    dual = dual1value(L, alpha, ones(k) / k, l2)
    print('Starting weight finding optimization for k=%d q=%d alpha=%f gap goal=%g' % (k, q, alpha, dualityGapGoal))
    for i in xrange(int(ceil(log(maxIters) + 1))):
        report(i, W, l2, None)

        print('Primal - dual = gap; %g - %g = %g' % (primal, dual, primal - dual))
        if primal - dual > dualityGapGoal:
            W = weightsForMultipleLosses2FISTA(L, alpha, W=W,
                                               maxIters=5 * (2 ** i),
                                               report=lambda Wk, ik: report(str(i) + 'P' + str(ik), Wk, None, None))
            primal = penalizedMultipleWeightedLoss2(L, W, alpha)
            l2 = dual1solve5(L, alpha, lambda2=l2, fixedLowerBound=-primal, maxIters=int(5 * (1.5 ** i)),
                             report=lambda l2, ik: report(str(i) + 'D' + str(ik), None, l2, None))
            dual = dual1value(L, alpha, ones(k) / k, l2)
        else:
            print('Achieved primal - dual = gap: %g - %g = %g' % (primal, dual, primal - dual))
            return W


def weightsForMultipleLosses2BlockMinimization(L, alpha, eta=None, W=None, maxIters=100, startAlpha=None, report=nop):
    ''' Solve a jointly convex problem of finding optimal weight vectors
    for multiple models with joint penalization by minimizing w.r.t. each
    weight vector iteratively (solving for a single weight vector is very
    efficient). Stop at a pretty rough tolerance.'''
    alpha = float(alpha)
    startAlpha = alpha if startAlpha is None else float(startAlpha)
    k, q = L.shape
    eta = ones(k) / k if eta is None else eta

    if W is not None:
        W = W.copy()
    else:
        #print('Received no warm start weights, choosing arbitrary.')
        W = ones((k, q)) / q

    oldW = ones((k, q))  # never close
    print('Starting joint weights problem at loss %f'
          % penalizedMultipleWeightedLoss2(L, W, alpha, eta))
    for i in arange(maxIters):
        theta = i / float(maxIters)
        currAlpha = (1 - theta) * startAlpha + theta * alpha
        report(W.copy(), i)
        oldW = W.copy()
        for j in arange(k):
            W[j, :] = adjustedWeightsForWeightedLosses(L, W, eta, alpha, j)
        if allclose(W, oldW, rtol=10 ** -7, atol=((1 / q) * 10 ** -7)):
            print('Change small at iteration %d, stopping early.' % i)
            break

    if i == maxIters - 1:
        print('Out of iterations, stopping.')
        pass

    return W


def weightsForMultipleLosses2RandomBlockMinimization(L, alpha, W=None, maxIters=100, startAlpha=None, report=nop):
    ''' Solve a jointly convex problem of finding optimal weight vectors
    for multiple models with joint penalization by minimizing w.r.t. each
    weight vector iteratively (solving for a single weight vector is very
    efficient). Stop at a pretty rough tolerance.'''
    alpha = float(alpha)
    startAlpha = alpha if startAlpha is None else float(startAlpha)
    k, q = L.shape
    if W is not None:
        W = W.copy()
    else:
        #print('Received no warm start weights, choosing arbitrary.')
        W = ones((k, q)) / q

    oldW = ones((k, q))  # never close
    print('Starting joint weights problem at loss %f'
          % penalizedMultipleWeightedLoss2(L, W, alpha))
    for i in arange(maxIters):
        theta = i / float(maxIters)
        currAlpha = (1 - theta) * startAlpha + theta * alpha
        report(W.copy(), i)
        oldW = W.copy()
        j = randint(k)
        allButCurrent = sum(W, 0) - W[j, :]
        h = ones(q) * k / q - allButCurrent
        newWeights = weightsForLosses(L[j, :], currAlpha / k, goal=h)
        W[j, :] = newWeights

        if allclose(W, oldW, rtol=10 ** -7, atol=((1 / q) * 10 ** -7)):
            print('Change small at iteration %d, stopping early.' % i)
            break

    if i == maxIters - 1:
        print('Out of iterations, stopping.')
        pass

    return W


def weightsForMultipleLosses2GradientProjection(L, alpha, eta=None, W=None, maxIters=200):
    k, q = L.shape
    eta = eta if eta is not None else ones(k) / k

    tk = alpha ** -1
    if W is None:
        W = ones((k, q)) / q
    for _ in range(maxIters):
        gk = penalizedMultipleWeightedLoss2Gradient(L, W, alpha, eta=eta)
        Wr = W - tk * gk.reshape(k, q)
        W = array([projectToSimplexNewton(w) for w in Wr])
        #print(penalizedMultipleWeightedLoss2(L, W, alpha, eta=eta))
    return W


def weightsForMultipleLosses2FISTA(L, alpha, eta, W=None, maxIters=200, report=nop, row_sums=None):
    k, q = L.shape
    if W is None:
        W = ones((k, q)) / q

    weightStream = weightsForMultipleLosses2FISTAStream(L, alpha, eta, W, row_sums=row_sums)

    return reportAndBoundStream(weightStream, maxIters=maxIters, report=report)


def tsByCuttingPlanes(L, alpha, eta, ts0=None, maxIters=100, report=nop):
    k, q = L.shape
    ts0 = ts0 if not ts0 is None else randn(k)

    tsStream = lowDimSolveStream(L, alpha, eta, ts0=ts0)

    return reportAndBoundStream(tsStream, maxIters=maxIters, report=report)



def weightsForMultipleLosses2FISTAStream(L, alpha, eta, W0, row_sums=None):
    k, q = L.shape
    row_sums = row_sums if not row_sums is None else ones(k)
    if W0 is None:
        W0 = ones((k, q)) / q

    f = lambda Wl: penalizedMultipleWeightedLoss2(L, Wl.reshape((k, q)), alpha, eta=eta)
    g = lambda Wl: 0 # Will be applied only to valid weights...
    grad_f = lambda Wl: penalizedMultipleWeightedLoss2Gradient(L, Wl, alpha, eta=eta)

    def prox_g(lip, old_W):
        newW = zeros_like(old_W)
        k, _ = newW.shape
        for j in range(k):
            projectToSimplexNewton(old_W[j, :], target=row_sums[j], into=newW[j, :])
        return newW

    return fastGradientProjectionStream(f, g, grad_f, prox_g, W0, initLip=alpha)


def penalizedMultipleWeightedLoss2Gradient(L, W, alpha, eta=None):
    k, q = L.shape
    eta = eta if eta is not None else ones(k) / k
    u = ones(q) / q

    lossesPart = (L.T * eta).T
    vMinusUniform = eta.dot(W) - u
    regPart = (2 * alpha * eta.reshape((k, 1))) * vMinusUniform.reshape((1, q))
    return regPart + lossesPart


def penalizedWeightedLoss(loss, weights, alpha):
    q = loss.shape[0]
    u = ones(q) / q
    dev = weights - u
    return loss.dot(weights) + alpha * dev.dot(dev)


def penalizedMultipleWeightedLoss(L, W, alpha):
    k, q = L.shape
    u = ones(q) / q
    dev = sum(W, 0) - u
    return sum(L * W) + alpha * dev.dot(dev)


def validateWeights(W):
    if sum(abs(W.sum(1) - 1)) > 1e-9 or (W < 0).any():
        raise Exception('Invalid weight vector!')


def penalizedMultipleWeightedLoss2(L, W, alpha, eta=None):
    k, q = L.shape
    eta = eta if eta is not None else ones(k) / k

    u = ones(q) / q
    if alpha <= 0:
        raise Exception('Alpha must be positive, not %s.' % alpha)
    dev = eta.dot(W) - u
    return sum(eta.dot(L * W)) + alpha * dev.dot(dev)


def dual1solve(L, alpha, lambda2=None, maxIters=200, rep=None, slow=False, proxVersion=None):
    ''' we use a stock minimization to maximaze dual1values w.r.t. lambda. '''
    alpha = float(alpha)
    k, q = L.shape
    if lambda2 is None:
        lambda2 = zeros(q)
    u = ones(q) / q

    def f(l2):
        v = vmin(alpha, l2)
        return -(alpha * (norm(u - v) ** 2) - l2.dot(v))

    def g(l2):
        return dual1g(L, l2)

    def gradf(l2):
        return vmin(alpha, l2)

    def proxg(Lip, l2):
        if proxVersion is None:
            return dual1prox2(Lip, L, l2)
        else:
            return proxVersion(Lip, L, l2)

    if slow:
        return slowGradientProjectionMethod(f, g, gradf, proxg, lambda2, maxIters=maxIters, report=rep)
    else:
        return fastGradientProjectionMethod(f, g, gradf, proxg, lambda2, maxIters=maxIters, report=rep,
                                            initLip=1. / alpha)


def weightsForMultipleLosses2dual1(L, alpha, maxIters=200, report=None):
    k, _ = L.shape
    l2 = dual1solve5(L, alpha, maxIters=maxIters, report=report)

    W = optimalWeightsMultipleModelsFixedProfile(L, vmin(alpha, l2))
    dualV = dual1value(L, alpha, ones(k) / k, l2)
    primV = penalizedMultipleWeightedLoss2(L, W, alpha)
    print('primal - dual = %f - %f = %f' % (primV, dualV, primV - dualV))
    return W


def dual1vw(L, alpha, lambda2):
    k, q = L.shape
    #Solve quadratic over simplex part.
    min_v = projectToSimplexNewton(lambda2 / 2 / alpha)
    wmins = array([(l + lambda2).min() for l in L])
    #wmins = (L+lambda2).min(1) # slower
    return min_v, wmins


def dual1g(L, l2):
    k, q = L.shape
    return - array([(l + l2 / k).min() for l in L]).sum()


def dual1value(L, alpha, eta, lambda2):
    u = ones_like(lambda2)
    u = u / u.sum()

    vmin, wmins = dual1vw(L, alpha, lambda2)
    return alpha * (norm(u - vmin) ** 2) - lambda2.dot(vmin) + eta.dot(wmins)


def dual1Q(Lip, L, newLambda, oldLambda):
    k, q = L.shape
    return Lip * (norm(newLambda - oldLambda) ** 2) / 2 + dual1g(L, newLambda)


def dual1Qsubgradient(L, newLambda, oldLambda):
    k, q = L.shape
    raise Exception('Function would return incorrect results')
    return newLambda - oldLambda - (uniformOnMinsPerRow(L + newLambda / k)).sum(0)


def uniformOnMinsPerRow(L):
    k, q = L.shape
    dist = zeros((k, q))
    for j in range(k):
        mins = L[j, :] == L[j, :].min()
        dist[j, mins] = 1. / mins.sum()
    return dist


def maybeProx(Lip, L, oldLambda):
    k, q = L.shape

    f = lambda t, modelIndex, l: dual1Q(Lip, L, l + k * nonNegativePart(t * ones(q) - (L[modelIndex, :] + l / k)),
                                        oldLambda)
    newLambda = oldLambda
    for _ in range(1):
        for j in range(k):
            lowerBound = (oldLambda / k + L[j, :]).min()
            upperBound = (oldLambda / k + L[j, :]).max() + 2
            optT = oneSidedBinarySearch(lambda t: f(t, j, newLambda), lowerBound)
            #optT = fminbound(lambda t: f(t,j,newLambda),lowerBound,upperBound,xtol=1e-13,disp=1)
            newLambda = newLambda + k * nonNegativePart(optT * ones(q) - (L[j, :] + newLambda / k))
    return newLambda


def maybeProx2(Lip, L, oldLambda):
    k, q = L.shape

    f = lambda t, modelIndex, l: dual1Q(Lip, L, l + k * nonNegativePart(t * ones(q) - (L[modelIndex, :] + l / k)),
                                        oldLambda)
    newLambda = oldLambda
    for _ in range(1):
        for j in range(k):
            lowerBound = (oldLambda / k + L[j, :]).min()
            upperBound = (oldLambda / k + L[j, :]).max() + 2
            #optT = oneSidedBinarySearch(lambda t: f(t,j,newLambda),lowerBound)
            optT = fminbound(lambda t: f(t, j, newLambda), lowerBound, upperBound, xtol=1e-13, disp=1)
            newLambda = newLambda + k * nonNegativePart(optT * ones(q) - (L[j, :] + newLambda / k))
    return newLambda


def altProx(Lip, L, oldLambda):
    k, q = L.shape
    o = ones(q)

    f = lambda ts: Lip * norm(oldLambda - lambdaForTsAndOld(L, ts, oldLambda)) ** 2 / 2 - sum(ts)

    currTs = zeros(k)
    for repeats in range(2):
        for h in range(k):
            currTs[h] = 0
            e = ei(h, k)
            lowerBound = (oldLambda / k + L[h, :]).min()
            upperBound = (oldLambda / k + L[h, :]).max() + 300
            currTs[h] = fminbound(lambda t: f(currTs + t * e),
                                  lowerBound,
                                  upperBound,
                                  xtol=1e-13)
    return lambda2FromTs(L, currTs)


def lambdaForTsAndOld(L, ts, oldLambda):
    k = ts.shape[0]
    return array([k * (ts[j] - L[j, :]) for j in range(k)] + [oldLambda]).max(0)


def altProx2(Lip, L, oldLambda):
    k, q = L.shape
    o = ones(q)
    f = lambda ts: Lip * norm(oldLambda - lambdaForTsAndOld(L, ts, oldLambda)) ** 2 / 2 - sum(ts)
    gradf = lambda ts: proxProxyGrad(Lip, L, oldLambda, ts)
    g = lambda ts: 0
    proxg = lambda Lip, ts: ts
    x0 = array([(oldLambda / k + l).min() for l in L])
    currTs = fastGradientProjectionMethod(f, g, gradf, proxg, x0, maxIters=5)
    #currTs = scipy.optimize.fmin_bfgs(f,zeros(k))
    return lambdaForTsAndOld(L, currTs, oldLambda)


def proxProxyGrad(Lip, L, oldLambda, ts):
    k, q = L.shape

    arr = array([ts[j] - L[j, :] for j in range(k)])
    idxes = list(arr.argmax(0))
    lambda2 = vstack((k * arr[idxes, range(q)], oldLambda)).max(0)
    diff = Lip * (lambda2 - oldLambda)
    locs = zeros((k, q))
    locs[idxes, range(q)] = 1
    return locs.dot(diff) * k - 1


def vmin(alpha, l2):
    return projectToSimplexNewton(l2 / 2 / alpha)

#@profile
def lambda2FromTs(L, ts):
    l, _ = lambdaAndWinners(L, ts)
    return l
    #((ts - L.T).T.max(0))


def tsFromLambda(L, l2):
    return (l2 + L).min(1)


#@profile
def gOfTs(L, alpha, ts, eta):
    k, q = L.shape
    u = ones(q) / q
    lambda2 = lambda2FromTs(L, ts)
    v = vmin(alpha, lambda2)
    return alpha * (norm(u - v) ** 2) - lambda2.dot(v) + eta.dot(ts)


#@profile
def gOfTGrad(L, alpha, ts, eta):
    k, q = L.shape
    u = ones(q) / q

    lambda2, idxes = lambdaAndWinners(L, ts)

    v = vmin(alpha, lambda2)
    dataPart = - array([v[idxes == j].sum() for j in range(k)])
    basicGradient = dataPart + eta
    return basicGradient - basicGradient.mean()


def lambda2W(L, l2, alpha, eta):
    return optimalWeightsMultipleModelsFixedProfile(L, vmin(alpha, l2), eta)


'''
def lambdaAndWinners(L, ts):
    arr = (ts - L.T).T

    # Find the best model to explain each point, and base weights there
    idxes = arr.argmax(0)
    maxes = arr.max(0)
    return idxes, maxes
'''


def ts2W(L, ts, alpha, eta):
    """ Create a weight matrix from losses and adjustment factors.

    In contrast to other methods, we use ts-L to partition the data points,
    then directly create valid weight distributions."""

    k, n = L.shape

    # Find the best model to explain each point, and base weights there
    lambda2, idxes = lambdaAndWinners(L, ts)
    kMaxes = lambda2 / 2. / alpha

    # Compute weights.
    W = zeros((k, n))
    for j in range(k):
        if sum(idxes == j) > 0:
            partial_wj = projectToSimplexNewton(kMaxes[idxes == j] / eta[j])
            W[j, idxes == j] = partial_wj
        else:
            W[j, :] = ones(n) / n
    return W


def ts2W2(L, ts, alpha, eta):
    """ Create a weight matrix from losses and adjustment factors.

    In contrast to other methods, we use ts-L to partition the data points,
    then directly create valid weight distributions."""

    k, n = L.shape

    # Find the best model to explain each point, and base weights there
    lambda2, idxes = lambdaAndWinners(L, ts)
    v = vmin(alpha, lambda2)

    # Compute weights.
    W = zeros((k, n))
    for j in range(k):
        jidx = where(idxes == j)[0]
        if sum(jidx) > 0:
            partial_wj = projectToSimplexNewton(v[jidx] / eta[j])
            W[j, jidx] = partial_wj
        else:
            W[j, :] = ones(n) / n
    return W


def ts2W3(L, ts, alpha, eta):
    """ Create a weight matrix from losses and adjustment factors.

    In contrast to other methods, we use ts-L to partition the data points,
    then directly create valid weight distributions."""

    k, n = L.shape

    # Find the best model to explain each point, and base weights there
    lambda2, idxes = lambdaAndWinners(L, ts)
    v = vmin(alpha, lambda2)

    # Compute weights.
    W = zeros((k, n))
    for j in range(k):
        jidx = where(idxes == j)[0]
        partial_v = v[jidx]
        assigned_weight = partial_v.sum()
        if assigned_weight > 0:
            partial_wj = partial_v / assigned_weight
            W[j, jidx] = partial_wj
        else:
            W[j, :] = ones(n) / n
    return W


def ts2WMixedColumns(L, ts, alpha, eta):
    """Create a weight matrix from losses and given adjustment factors.

    An important subtlety is that only for the optimal ts do we know that
    the resulting weight rows sum to 1."""
    k, n = L.shape
    etai = eta ** -1

    lambda2 = lambda2FromTs(L, ts)

    # Compute the adjusted fit
    arr = array([ts[j] - L[j, :] for j in range(k)])

    # Find the best model to explain each point (not final)
    idxes = arr.argmax(0)
    etavi = etai[idxes]

    v = vmin(alpha, lambda2)

    # Set the easy parts of W
    W = zeros((k, n))

    # Detect near ties
    maxes = arr.max(0)
    co_maximizers = abs(arr - maxes) < 0.00001
    ambiguous = (co_maximizers.sum(0) > 1)
    ambPos = ambiguous.nonzero()[0]
    nonAmbPos = (1 - ambiguous).nonzero()[0]

    # Set W for the non-ambiguous set of points
    W[idxes[nonAmbPos], nonAmbPos] = etavi[nonAmbPos] * v[nonAmbPos]
    if ambiguous.any():
        # Below red stands for "reduced", corresponding to the ambiguous columns/points.
        redL = L[:, ambPos]
        redV = v[ambPos]
        rowSums = W[:, nonAmbPos].sum(1)
        rawRedRowSums = 1 - rowSums
        redRowSums = projectToSimplexNewton(rawRedRowSums,
                                            target=rawRedRowSums.sum())
        if (redRowSums < 0).any():
            pdb.set_trace()
        redW = optimalWeightsMultipleModelsFixedProfile(redL, redV, eta, rowSums=redRowSums)
        # Combine
        W[:, ambiguous] = redW

    return array([projectToSimplexNewton(w) for w in W])


def ts2WMixedColumnsCorrectedSupport(L, ts, alpha):
    """Create a weight matrix from losses and given adjustment factors.

    An important subtlety is that only for the optimal ts do we know that
    the resulting weight rows sum to 1."""
    k, n = L.shape
    lambda2 = lambda2FromTs(L, ts)

    # Compute the adjusted fit
    #arr = array([ts[j] - L[j, :] for j in range(k)])
    arr = (ts - L.T).T
    # Find the best model to explain each point (not final)
    idxes = arr.argmax(0)

    v = vmin(alpha, lambda2)

    # Set the easy parts of W
    W = zeros((k, n))

    # Detect near ties
    maxes = arr.max(0)
    co_maximizers = abs(arr - maxes) < 0.00001
    ambiguous = (co_maximizers.sum(0) > 1)
    ambPos = ambiguous.nonzero()[0]
    nonAmbPos = (1 - ambiguous).nonzero()[0]

    # Set W for the non-ambiguous set of points
    W[idxes[nonAmbPos], nonAmbPos] = k * v[nonAmbPos]
    if ambiguous.any():
        # Below red stands for "reduced", corresponding to the ambiguous columns/points.
        redL = L[:, ambPos]
        redV = v[ambPos]
        rowSums = W[:, nonAmbPos].sum(1)
        rawRedRowSums = 1 - rowSums
        redRowSums = projectToSimplexNewton(rawRedRowSums,
                                            target=rawRedRowSums.sum())
        if (redRowSums < 0).any():
            pdb.set_trace()
        redW = optimalWeightsMultipleModelsFixedProfile(redL, redV, rowSums=redRowSums)
        # Combine
        W[:, ambiguous] = redW
    """
    W = zeros((k,n))
    for j in range(k):
        active = W[j,:] > 0
        if sum(active) > 0:
            W[j, active] = projectToSimplexNewton(W[j, active])
        else:
            W[j,:] = ones(n) / n"""

    return array([w / sum(w) for w in W])


def ts2WFista(L, ts, alpha, eta):
    """Create a weight matrix from losses and given adjustment factors.

    An important subtlety is that only for the optimal ts do we know that
    the resulting weight rows sum to 1."""
    k, n = L.shape
    W0 = ts2W(L, ts, alpha, eta)

    return weightsForMultipleLosses2FISTA(L, alpha, eta, W=W0, maxIters=10)


def t2wExperimental(L, ts):
    k, q = L.shape
    return [projectToSimplexNewton(ts[j] - L[j, :]) for j in range(k)]


def sparsityUsingLambda2W(L, l2, alpha):
    k, q = L.shape
    ts = tsFromLambda(L, l2)
    adaptiveFudge = sort(((l2 / k + L).T - ts).flatten())[q + 10 * k * k]
    A = ((l2 / k + L).T - ts).T <= adaptiveFudge
    ambIdx = A.sum(0) > 1
    v = vmin(alpha, l2)
    W = A * k * v
    if ambIdx.sum() > 0:
        allocated = W[:, ambIdx < 1].sum(1)
        Wamb = optimalWeightsMultipleModelsFixedProfile(L[:, ambIdx], v[ambIdx], rowSums=1 - allocated)
        W[:, ambIdx] = Wamb
    return W


def heuristicL2ToW(L, l2, alpha, fudge=1e-15):
    k, q = L.shape
    ts = tsFromLambda(L, l2)
    A = ((l2 / k + L).T - ts).T <= fudge
    # Each column of B is for a single point either a distribution over
    # models (how to split weight) or a zero vector (if none is due)
    B = nan_to_num((1. * A) / (A.sum(0)))
    v = vmin(alpha, l2)
    return array([w / sum(w) for w in B * v * k])


def heuristicL2ToWproj(L, l2, alpha, fudge=1e-15):
    k, q = L.shape
    ts = tsFromLambda(L, l2)
    A = ((l2 / k + L).T - ts).T <= fudge
    # Each column of B is for a single point either a distribution over
    # models (how to split weight) or a zero vector (if none is due)
    B = nan_to_num((1. * A) / (A.sum(0)))
    v = vmin(alpha, l2)
    return array([projectToSimplexNewton(w + mask) for (w, mask) in zip(B * v * k, A)])


def heuristic2L2ToW(L, l2, alpha, fudge=1e-15):
    k, q = L.shape
    ts = tsFromLambda(L, l2)
    A = ((l2 / k + L).T - ts).T <= fudge
    'Each column of B is for a single point either a distribution over models (how to split weight) or a zero vector (if none is due)'
    B = nan_to_num((1. * A) / (A.sum(0)))
    v = vmin(alpha, l2)
    W0 = array([projectToSimplexNewton(w + mask) for (w, mask) in zip(B * v * k, A)])
    W1 = array([vi * k * nan_to_num(projectToSimplexNewton(nan_to_num(col / vi / k) + Acol))
                for col, Acol, vi
                in zip(W0.T, A.T, v)]).T
    return array([projectToSimplexNewton(w + mask) for (w, mask) in zip(W1, A)])


def dual1solve5(L, alpha, eta=None, fixedLowerBound=None, lambda2=None, maxIters=30, report=None):
    k, _ = L.shape
    eta = eta if eta is not None else ones(k) / k
    tStream = dual1solve5Stream(L, alpha, eta=eta, fixedLowerBound=fixedLowerBound, lambda2=lambda2)

    return reportAndBoundStream(tStream,
                                maxIters=maxIters,
                                report=report)


#@profile
def dual1solve5Stream(L, alpha, eta=None, fixedLowerBound=None, lambda2=None):
    k, q = L.shape
    lambda2 = zeros(q) if lambda2 is None else lambda2

    nextLowerBound = fixedLowerBound
    tStream = proximalCuttingPlaneStream(lambda t: -gOfTs(L, alpha, t, eta),
                                         lambda t: -gOfTGrad(L, alpha, t, eta),
                                         tsFromLambda(L, lambda2),
                                         lowerBound=nextLowerBound,
                                         stepSize=200.)
    ts = tStream.next()
    while True:
        lambda2 = lambda2FromTs(L, ts)
        nextLowerBound = yield lambda2
        ts = tStream.send(nextLowerBound)


def lowDimSGDStream(L, alpha, eta, subsamplingFactor, radius, ts0=None):
    k, n = L.shape
    sampleSize = ceil(subsamplingFactor * n)
    ts0 = randn(k) if ts0 is None else ts0

    def sGOfTGrad(t):
        return -gOfTGrad(subset(L, sampleSize),
                         alpha * sampleSize / n,
                         t, eta)

    def emptyProj(t):
        return t

    oneOverN = (radius * 50. / (i + 50) for i in count(1))
    oneOverSqrtN = (radius * 10. / sqrt(i + 100) for i in count(1))

    unregProx = lambda g, C: -g / C
    return sgdStream(repeat(sGOfTGrad), ts0, oneOverN)
    #    return regularizedDualAveragingStream(repeat(sGOfTGrad), unregProx, ts0, lip, gamma)

#projectedSubgradientStream(sGOfTGrad, emptyProj, ts0, theta=sqrt(k)*(L.max()-L.min()))


def lowDimSolveStream(L, alpha, eta, fixedLowerBound=None, ts0=None):
    k, q = L.shape
    ts0 = randn(k) if ts0 is None else ts0
    ts0 = ts0 - ts0.mean()
    nextLowerBound = fixedLowerBound
    tStream = proximalCuttingPlaneStream(lambda t: -gOfTs(L, alpha, t, eta),
                                         lambda t: -gOfTGrad(L, alpha, t, eta),
                                         ts0,
                                         lowerBound=nextLowerBound,
                                         stepSize=1000.)
    ts = tStream.next()
    while True:
        nextLowerBound = yield ts
        ts = tStream.send(None) # primal lower bounds are generally very loose.
        #ts = tStream.send(nextLowerBound)


def lowDimCoordinateMinimizationStream(L, alpha, ts0=None):
    k, _ = L.shape
    ts0 = randn(k) if ts0 is None else ts0
    ts0 = ts0 - ts0.mean()

    b = L.max()

    while True:
        for j in range(k):
            tst = ts.copy()

            def f(tj):
                tst[j] = tj
                return -gOfTs(L, alpha, tst)

            ts[j] = fminbound(f, -b, b, xtol=1e-10)
            ts = ts - ts.mean()
            b = 2 * abs(ts).max()
            yield ts


def lowDimCoordinateMinimizationStream2(L, alpha, ts0=None):
    k, _ = L.shape
    ts0 = randn(k) if ts0 is None else ts0
    ts = ts0 - ts0.mean()
    #ts = zeros(k) if ts is None else ts
    b = L.max()

    modvecs = modVecs(k)
    while True:
        #j = sumWeightsFor(alpha, L,ts).argmin()
        #j = randint(k)
        j = ((sumWeightsFor(alpha, L, ts) - 1) ** 2).argmax()

        def f(tj):
            return sumWeightLoss(alpha, L, ts + tj * modvecs[j], j)

        ts += modvecs[j] * fminbound(f, -b, b, xtol=1e-5)
        ts = ts - ts.mean()
        #print "ts %s, j %s" % (ts,j)
        #print "sums: %s" % sumWeightsFor(alpha, L, ts)
        #if abs(sumWeightsFor(alpha, L, ts)[j] - 1) > 0.01:
        #    pdb.set_trace()
        b = 2 * abs(ts).max()
        yield ts


def lowDimCoordinateMinimizationStream3(L, alpha, ts0=None):
    k, _ = L.shape
    ts0 = randn(k) if ts0 is None else ts0
    ts = ts0 - ts0.mean()

    #ts = zeros(k) if ts is None else ts
    b = L.max()

    modvecs = modVecs(k)
    while True:
        j = ((sumWeightsFor(alpha, L, ts) - 1) ** 2).argmax()

        def f(tj):
            return sumWeightLoss(alpha, L, ts + tj * modvecs[j], j)

        ts += modvecs[j] * fminbound(f, -b, b, xtol=1e-5)
        ts = ts - ts.mean()
        print "ts %s, j %s" % (ts, j)
        print "sums: %s" % sumWeightsFor(alpha, L, ts)
        #if abs(sumWeightsFor(alpha, L, ts)[j] - 1) > 0.01:
        #    pdb.set_trace()
        b = 2 * abs(ts).max()
        yield ts


def modVecs(k):
    kt = float(k)
    return diag(ones(kt) * (kt / (kt - 1))) - ones((kt, kt)) / (kt - 1)


def weightsFor(alpha, L, ts, a):
    k, n = L.shape
    arr = array([ts[j] - L[j, :] for j in range(k)])
    idxes = arr.argmax(0)
    zn = zeros(n)
    W = zeros_like(L)
    for j in range(k):
        W[j, idxes == j] = k * (arr[j, idxes == j] / 2 / alpha + a)
        W[j, :] = vstack((W[j, :], zn)).max(0)
    return W


def sumWeightsFor(alpha, L, ts):
    a = aForTs(alpha, L, ts)
    return weightsFor(alpha, L, ts, a).sum(1)


def sumWeightLoss(alpha, L, ts, j):
    return (sumWeightsFor(alpha, L, ts)[j] - 1) ** 2


def aForTs(alpha, L, ts):
    lambda2 = lambda2FromTs(L, ts)
    v = vmin(alpha, lambda2)
    return median(v - lambda2 / (2 * alpha))


def dualPrimalCoordinateWise(L, alpha, eta, ts0=None):
    k, n = L.shape
    ts = randn(k) if ts0 is None else ts0
    ts = ts - ts.mean()
    yield ts
    while True:
        js_perm = permutation(k)
        for j in js_perm:
            v = vmin(alpha, lambda2FromTs(L, ts))
            js = [j_ for j_ in range(k) if j_ is not j]
            d = lambda2FromTs(L[js, :], ts[js]) + L[j, :]
            srt = d.argsort()
            d = d[srt]
            v_sum = cumsum(v[srt])
            fi = (v_sum >= eta[j]).nonzero()[0][0]
            ts[j] = d[fi]
            # The following seems to work ok, but doesn't seem correct mathematically...
            '''if fi > 0:
                ratio = (eta[j] - v_sum[fi - 1]) / (v_sum[fi] - v_sum[fi - 1])
                ts[j] = d[fi - 1] + (d[fi] - d[fi - 1]) * ratio
            else:
                ts[j] = d[fi - 1]'''
        yield ts


def dualCoordinateWiseStream(L, alpha, eta, ts0=None):
    k, n = L.shape
    ts = randn(k) if ts0 is None else ts0
    ts = ts - ts.mean()
    c = LambdaAndWinnersTree(L, ts.copy())

    yield ts.copy()
    yield ts.copy()
    yield ts.copy()
    while True:
        js_perm = permutation(k)
        for j in js_perm:
            l2, win = c.lambda_and_winners()
            v = vmin(alpha, l2)
            l2nj, _ = c.lambda_and_winners_all_but(j)
            d = l2nj + L[j, :]
            srt = d.argsort()
            d = d[srt]
            v_sum = cumsum(v[srt])
            fi = (v_sum >= eta[j]).nonzero()[0][0]
            ts[j] = d[fi]
            # The following seems to work ok, but doesn't seem correct mathematically...
            '''if fi > 0:
                ratio = (eta[j] - v_sum[fi - 1]) / (v_sum[fi] - v_sum[fi - 1])
                ts[j] = d[fi - 1] + (d[fi] - d[fi - 1]) * ratio
            else:
                ts[j] = d[fi - 1]'''
            c.update_t_at(j, ts[j])
        yield ts.copy()

#@profile
def dualPrimalCoordinateWise(L, alpha, eta, ts0=None, maxIters=1000, dualityGapGoal=1e-6, report=nop):
    tsStr = dualPrimalCoordinateWiseStream(L, alpha, eta, ts0=ts0)

    ts = tsStr.next()
    report('dc-', None, None, ts.copy())

    i = 0
    while i < maxIters:
        i += 1
        W = ts2W(L, ts, alpha, eta)
        primal_value = penalizedMultipleWeightedLoss2(L, W, alpha, eta)
        report('p1' + str(i), W.copy(), None, None)
        ts = tsStr.next()
        report('dc' + str(i), None, None, ts.copy())

        dual_value = gOfTs(L, alpha, ts, eta)
        if primal_value - dual_value < dualityGapGoal:
            break


def dualPrimalCoordinateWiseStream(L, alpha, eta, ts0=None):
    k, n = L.shape
    ts = randn(k) if ts0 is None else ts0
    ts = ts - ts.mean()
    yield ts.copy()
    c = LambdaAndWinnersTree(L, ts.copy())
    prev_dual = gOfTs(L, alpha, ts, eta)

    while True:
        js_perm = permutation(k)
        for j in js_perm:
            # compute a v
            l2, win = c.lambda_and_winners()
            v = vmin(alpha, l2)

            # compute disadvantage of current model, sort by it
            l2nj, _ = c.lambda_and_winners_all_but(j)
            d = l2nj + L[j, :]
            srt = d.argsort()
            d = d[srt]

            # find t to cancel advantage at correct cumulative v
            v_sum = cumsum(v[srt])
            fi = (v_sum >= eta[j]).nonzero()[0][0]
            delta_ts_j = d[fi] - ts[j]

            nts = ts.copy()
            nts[j] = ts[j] + delta_ts_j
            new_dual = gOfTsJVsRest(L, alpha, nts, eta, j, l2nj)
            #print "test correctness"
            if new_dual < prev_dual:
                #pdb.set_trace()
                def to_min(delta_tsj):
                    nts[j] = ts[j] + delta_tsj
                    return -gOfTsJVsRest(L, alpha, nts, eta, j, l2nj)
                    #return -gOfTsJVsRest(L, alpha, mts, eta, j, l2nj)
                delta_ts_j = fminbound(to_min, min(-delta_ts_j, delta_ts_j), max(-delta_ts_j, delta_ts_j), xtol=1e-2)
                nts[j] = ts[j] + delta_ts_j
                prop_dual = gOfTs(L, alpha, nts, eta)
                print("Doing j: %i Prev: %f proposed: %f resulting: %f" % (j, prev_dual, new_dual, prop_dual))
                new_dual = prop_dual

            prev_dual = new_dual
            ts[j] = nts[j]
            '''# debugging
            old_dual_val = gOfTs(L, alpha, ts, eta)
            new_dual_val = gOfTs(L, alpha, nts, eta)
            if old_dual_val > new_dual_val:
                print nts - ts
                pdb.set_trace()
            ts[j] = d[fi]'''

            c.update_t_at(j, ts[j])
            yield ts.copy()


def gOfTsJVsRest(L, alpha, ts, eta, j, lambda_all_but_j):
    k, q = L.shape
    u = ones(q) / q
    lj = ts[j] - L[j, :]
    lambda2 = maximum(lambda_all_but_j, lj)
    v = vmin(alpha, lambda2)
    return alpha * (norm(u - v) ** 2) - lambda2.dot(v) + eta.dot(ts)
