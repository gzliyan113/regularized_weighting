import time
from numpy.random import rand
from numpy import ones, array, cumsum, nan, isnan, zeros, zeros_like, ceil, median, abs, sqrt
import pdb

from matplotlib.pyplot import semilogy, show, legend, xlabel, ylabel, title
from pandas import Series, DataFrame
from minimize import averageLateWeightingStream
from utility import projectToSimplexNewton, subset, reportAndBoundStream
import minL2PenalizedLossOverSimplex as mp
from minL2PenalizedLossOverSimplex import heuristicL2ToW, heuristic2L2ToW, heuristicL2ToWproj, tsFromLambda, lambda2FromTs, lowDimSolveStream
import cvxoptJointWeightsOptimization as co
#import quadraticRegressionWeightAssignmentBenchmarks as qbench
#from memory_profiler import profile

def timeAndComputeLoss(func, params, L):
    startt = time.time()
    rawW = func(L, alpha, **params)
    endt = time.time()
    W = array([projectToSimplexNewton(w) for w in rawW])
    loss = mp.penalizedMultipleWeightedLoss2(L, W, alpha)
    return endt - startt, loss

#@profile
def validateScoreWeights(alpha, L, W, eta):
    k,_ = L.shape
    if W is None:
        return None
    sums = W.sum(axis=1)
    if (abs(sums-ones(k)) > 0.001).any():
        return 10e20
    return mp.penalizedMultipleWeightedLoss2(L, W, alpha, eta)

#@profile
def recordIntermediateLosses(func, params, L, alpha, eta):
    seq = []
    def rep(id, W, lambda2, ts):
        t = time.time()
        seq.append((id, t, validateScoreWeights(alpha, L, W, eta), lambda2, ts))

    starttime = time.time()
    func(L, alpha, eta, report=rep, **params)
    endtime = time.time()
    ids, times, validatedWScores, lambdas, ts = zip(*seq)

    #Ws = ((array([projectToSimplexNewton(w) for w in rawW]) if not rawW is None else None) for rawW in rawWs)
    ts = Series(ts, index=ids).fillna(method='pad')
    if (not ts.isnull().all()):
        pass #pdb.set_trace() if (ts[ts.notnull()][0].shape[0] > 2) else 3

    effLambdas = Series(
        [(lambda2FromTs(L, t) if not t is None else nan) for t in ts] if not ts.isnull().all() else lambdas, index=ids)

    return (Series(array(times) - array([starttime] + list(times[:-1])), index=ids),
            #Series(array([(mp.penalizedMultipleWeightedLoss2(L, W, alpha) if not W is None else nan) for W in Ws]),
            Series(array([(ws if not ws is None else nan) for ws in validatedWScores]),
                   index=ids).fillna(method='pad'),
            Series(array([(mp.dual1value(L, alpha, eta, l2) if (not l2 is None) and not isnan(l2).any() else nan)
                          for l2 in effLambdas]), index=ids).fillna(method='pad'))


def makePrimalToFullWrapper(func):
    def primalWrapper(L, alpha, eta, report=None, **params):
        func(L, alpha, eta, report=lambda xk, k: report(k, xk, None, None), **params)

    return primalWrapper


def makeDualTsToFullWrapper(func):
    def dualTsWrapper(L, alpha, report=None, **params):
        func(L, alpha, report=lambda ts, k: report(k, None, None, ts), **params)

    return dualTsWrapper


def makeDualLambdasToFullWrapper(func):
    def dualLambdasWrapper(L, alpha, report=None, **params):
        func(L, alpha, report=lambda l2, k: report(k, None, l2, None), **params)

    return dualLambdasWrapper


def makePrimalToFullRepWrapper(report, prefix=None):
    def primalRepWrapper(W, k):
        report(k if prefix is None else prefix + str(k), W, None, None)

    return primalRepWrapper


def makeDualTsToFullRepWrapper(report, prefix=None):
    def dualTsRepWrapper(ts, k):
        report(k if prefix is None else prefix + str(k), None, None, ts)

    return dualTsRepWrapper


def linearWrapper(L, alpha):
    k, q = L.shape
    return co.optimalWeightsMultipleModelsFixedProfile(L, ones(q) / q)


def linThenFISTA(L, alpha, maxIters):
    k, q = L.shape
    W0 = co.optimalWeightsMultipleModelsFixedProfile(L, ones(q) / q)
    return mp.weightsForMultipleLosses2FISTA(L, alpha, W=W0, maxIters=maxIters)


def guesses(L, alpha, maxIters):
    k, q = L.shape
    lambda2 = k * ((-L).max(0))
    return heuristicL2ToW(L, lambda2, alpha)


def guesses3(L, alpha, maxIters):
    k, q = L.shape
    lambda2 = k * ((-L).max(0))
    return heuristicL2ToWproj(L, lambda2, alpha)


def guesses2(L, alpha, maxIters):
    k, q = L.shape
    lambda2 = k * ((-L).max(0))
    return heuristic2L2ToW(L, lambda2, alpha)


def guessesThenFISTA(L, alpha, maxIters, report):
    ''' Simple guess at lambda2, then compute ts, refine lambda2 by optimality conditions'''
    W0 = guesses(L, alpha, maxIters)
    return mp.weightsForMultipleLosses2FISTA(L, alpha, W=W0, maxIters=maxIters, report=report)


def guessesThenBlock(L, alpha, maxIters, report):
    ''' Simple guess at lambda2, then compute ts, refine lambda2 by optimality conditions'''
    W0 = guesses(L, alpha, maxIters)
    report(W0, 0)

    def rep(W, k):
        report(W, k + 1)

    return mp.weightsForMultipleLosses2BlockMinimization(L, alpha, W=W0, maxIters=maxIters, report=rep)


def linFISTALin(L, alpha, maxIters):
    k, q = L.shape
    W0 = co.optimalWeightsMultipleModelsFixedProfile(L, ones(q) / q)
    W1 = mp.weightsForMultipleLosses2FISTA(L, alpha, W=W0, maxIters=maxIters)
    return co.optimalWeightsMultipleModelsFixedProfile(L, W1.mean(0))


def linThenBlockMin(L, alpha, maxIters):
    k, q = L.shape
    W0 = co.optimalWeightsMultipleModelsFixedProfile(L, ones(q) / q)
    return mp.weightsForMultipleLosses2BlockMinimization(L, alpha, W=W0, maxIters=maxIters)


def dualThenHeuristicConversion(L, alpha, maxIters):
    k, q = L.shape
    l2 = mp.dual1solve4(L, alpha, maxIters=maxIters)
    ts = (l2 / k + L).min(1)
    l2sharp = k * ((ts - L.T).T.max(0))
    return heuristicL2ToW(L, l2sharp, alpha)


def dualThenExactConversion(L, alpha, maxIters):
    k, q = L.shape
    l2 = mp.dual1solve4(L, alpha, maxIters=maxIters)
    ts = (l2 / k + L).min(1)
    l2sharp = k * ((ts - L.T).T.max(0))
    return heuristicL2ToW(L, l2sharp, alpha)


def dualThenHeuristic2Conversion(L, alpha, maxIters):
    k, q = L.shape
    l2 = mp.dual1solve4(L, alpha, maxIters=maxIters)
    ts = tsFroml2(l2, k, L)
    l2sharp = k * ((ts - L.T).T.max(0))
    return heuristic2L2ToW(L, l2sharp, alpha)


def dual5ThenReducedLinear(L, alpha, maxIters):
    l2 = mp.dual1solve5(L, alpha, maxIters=maxIters)
    ts = tsFroml2(l2, k, L)
    l2sharp = k * ((ts - L.T).T.max(0))
    return mp.sparsityUsingLambda2W(L, l2sharp, alpha)


def dual5ThenLinear(L, alpha, eta, maxIters, report=None):
    k, q = L.shape
    report('d0', None, zeros(q), None)
    report('p0', zeros_like(L), None, None)
    l2 = mp.dual1solve5(L, alpha, maxIters=maxIters)
    report('d1', None, l2, None)
    print 'Starting linear completion of W'
    W0 = mp.lambda2W(L, l2, alpha, eta)
    print 'Done with linear completion of W'
    report('p1', W0, None, None)
    return W0


def dual5ThenCheaperLinear(L, alpha, maxIters, report=None):
    k, q = L.shape
    report('d0', None, zeros(q), None)
    report('p0', zeros_like(L), None, None)
    #    ts = reportAndBoundStream(
    #      lowDimSolveStream(L, alpha, ts=zeros(k)),
    #      maxIters=maxIters,
    #      report=report)
    l2 = mp.dual1solve5(L, alpha, maxIters=maxIters)
    report('d1', None, l2, None)
    ts = tsFromLambda(L, l2)
    report('dc', None, None, ts)
    print 'Starting linear completion of W'
    W0 = mp.ts2WMixedColumns(L, ts, alpha)
    print 'Done with linear completion of W'
    report('p1', W0, None, None)
    return W0


def noisyDualPrimal(L, alpha, maxIters, report=None):
    ratio = 0.1
    Lorig = L
    L = subset(Lorig, ceil(Lorig.shape[1]*ratio))
    lowDimDualStr = mp.lowDimSolveStream(L, alpha)
    initTs = lowDimDualStr.next()
    #W = mp.ts2WMixedColumns(Lorig, initTs, alpha)
    #primalValue = mp.penalizedMultipleWeightedLoss2(L, W, alpha)
    for i in xrange(maxIters):
        for h in xrange(10):
            ts = lowDimDualStr.next() #send(-primalValue)
            report('dca' + str(i) + str(h), None, None, ts)
        W = mp.ts2W(Lorig, ts, alpha)
        report('p1' + str(i), W, None, None)
        #primalValue = mp.penalizedMultipleWeightedLoss2(L, W, alpha)


def sgdDualPrimal(L, alpha, eta, maxIters, report=None):
    lowDimDualStr = mp.lowDimSGDStream(L, alpha, eta, 0.01, 1.)
    initTs = lowDimDualStr.next()
    #W = mp.ts2WMixedColumns(Lorig, initTs, alpha)
    for i in xrange(maxIters):
        for h in xrange(30):
            ts = lowDimDualStr.next() #send(-primalValue)
        report('dcs' + str(i), None, None, ts)
        W = mp.ts2W(L, ts, alpha, eta)
        report('p1' + str(i), W, None, None)


def sgdDualPrimal2(L, alpha, eta, maxIters, report=None):
    ratio = 0.01
    lowDimDualStr = averageLateWeightingStream(mp.lowDimSGDStream(L, alpha, eta, ratio, 1.))
    initTs = lowDimDualStr.next()
    #W = mp.ts2WMixedColumns(Lorig, initTs, alpha)
    for i in xrange(maxIters):
        for h in xrange(30):
            ts = lowDimDualStr.next() #send(-primalValue)
        report('dcg' + str(i), None, None, ts)
        W = mp.ts2W(L, ts, alpha, eta)
        report('p1' + str(i), W, None, None)



def dual5ThenCheaperLinear(L, alpha, maxIters, report=None):
    k, q = L.shape
    report('d0', None, zeros(q), None)
    report('p0', zeros_like(L), None, None)
    #    ts = reportAndBoundStream(
    #      lowDimSolveStream(L, alpha, ts=zeros(k)),
    #      maxIters=maxIters,
    #      report=report)
    #l2 = mp.dual1solve5(L, alpha, maxIters=maxIters)
    ts = reportAndBoundStream(lowDimSolveStream(L, alpha), maxIters=maxIters, report=None)
    l2 = lambda2FromTs(L, ts)
    report('d1', None, l2, None)
    ts = tsFromLambda(L, l2)
    report('dc', None, None, ts)
    print 'Starting linear completion of W'
    W0 = mp.ts2WMixedColumns(L, ts, alpha)
    print 'Done with linear completion of W'
    report('p1', W0, None, None)
    return W0


def dual5ThenHack(L, alpha, maxIters):
    l2 = mp.dual1solve5(L, alpha, maxIters=maxIters)
    ts = (l2 / k + L).min(1)
    return mp.t2wExperimental(L, ts)


def guessesDualThenHeuristicConversion(L, alpha, maxIters, report=None):
    k, q = L.shape
    W0 = guesses(L, alpha, maxIters)
    report('g0', W0, None, None)
    l2 = mp.dual1solve5(L, alpha, fixedLowerBound=-mp.penalizedMultipleWeightedLoss2(L, W0, alpha), maxIters=maxIters,
                        report=lambda l2, k: report(k, None, l2, None))
    #pdb.set_trace()
    W1 = heuristicL2ToW(L, l2, alpha)
    report(maxIters + 1, W1, None, None)
    return W1


def guessesDualThenFISTA(L, alpha, maxIters, report=None):
    W1 = guessesDualThenHeuristicConversion(L, alpha, maxIters, report=report)
    return mp.weightsForMultipleLosses2FISTA(L, alpha, W=W1, maxIters=maxIters,
                                             report=makePrimalToFullRepWrapper(report, prefix='F'))


def timesAndLossesReport():
    timesAndLosses = [timeAndComputeLoss(func, params, L) for (func, params, name) in algsAndNames]

    minLoss = min([l for (t, l) in timesAndLosses])

    for ((alg, params, name), (runTimeMS, loss)) in zip(algsAndNames, timesAndLosses):
        print('Alg: %s Took %g sec, has excess loss of at least %g' % (name, runTimeMS, loss - minLoss))


def asRange(a, b):
    a = b if a is None else a
    b = a if b is None else b
    return min(a, b), max(a, b)



k = 2
q = 10000

L = rand(k, q)
alpha = 5.*q
eta = ones(k) / k + rand(k)
eta = eta / eta.sum()
'''
q=1000
seedUsed, L, data = qbench.twoQuadraticsOnLineFatTailedNoiseCreateDataAndRandomModelLosses(q,1)
k,q = L.shape
alpha = median(L) * q
'''

print('Initialized data: k,q,alpha=%s,%s,%s' % (k,q,alpha))

wrappedFista = makePrimalToFullWrapper(mp.weightsForMultipleLosses2FISTA)
#stdp = {'maxIters': 50, 'ts': rand(k)}
#longp = {'maxIters': 200, 'ts': rand(k)}
stdp = {'maxIters': 6}
longp = {'maxIters': 100}
algSpec = [
    # (dual5ThenCheaperLinear, 'iterations of dual5 then linear via kxk conversion.', stdp),
    #(mp.weightsForMultipleLosses2DualPrimal, 'dual-primal via kxk conversion.', longp),
    #(mp.weightsForMultipleLosses2DualPrimal3c, 'new dual-primal via t-L conversion, with cache.', stdp),
    (mp.weightsForMultipleLosses2DualPrimal2, 'dual-primal via cutting plane w. t-L partition conversion.', longp),
    (mp.dualPrimalCoordinateWise, 'dual-primal via coord-wise w. t-L partition conversion.', stdp),

    #(mp.weightsForMultipleLosses2DualPrimal3, 'new dual-primal via t-L conversion.', stdp),
    #(mp.weightsForMultipleLosses2DualPrimal4, 'dual-primal via changed support correction.', stdp),
#    (mp.weightsForMultipleLosses2DualPrimal5, 'dual-primal on CA + changed support correction.', stdp),
#    (mp.weightsForMultipleLosses2DualPrimal6, 'dual-primal on C sum weight search + changed support correction.', stdp),
    #(dual5ThenLinear, 'iterations of dual5 then linear.', stdp),
#(noisyDualPrimal,'DualPrimal algorithm on 1/3rd of L.',stdp),
    #(sgdDualPrimal,'sgd DualPrimal algorithm.',stdp),
    #(sgdDualPrimal2,'sgd DualPrimal algorithm, with late averaging.',stdp),
    (wrappedFista,'FISTA baseline.',stdp),
    #(makeDualLambdasToFullWrapper(mp.dual1solve5), 'ProxCutPlane dual baseline.', stdp),
#(makeDualTsToFullWrapper(mp.dual1solve2), 'FISTA on dual (because it sometimes works to a point).', stdp),
    #(makePrimalToFullWrapper(mp.weightsForMultipleLosses2BlockMinimization), 'Blockwise minimization, round robins.', stdp),
    #    (makePrimalToFullWrapper(guessesThenBlock), 'Guess1 then Blockwise', stdp),
    #    (mp.primalDual, 'Primal-dual with fista optimization and prox-cutting-plane method.', longp),

    #    (makePrimalToFullWrapper(mp.weightsForMultipleLosses2BlockMinimization), 'Blockwise minimization, with changing alpha, 100 round robins.', {'maxIters':300, 'startAlpha': alpha / 2.}),
    #(mp.weightsForMultipleLosses2OldPrimalDual,
    # 'Mixed primal dual (essentially gradient descent on each, where latter is non-smooth)', longp)
]

tpd, descs = zip(*[(recordIntermediateLosses(alg, p, L, alpha, eta), desc) for alg, desc, p in algSpec])
times, primals, duals = zip(*tpd)

primalFrame = DataFrame(dict(zip(descs, primals)))
dualFrame = DataFrame(dict(zip(descs, duals)))
minP = primalFrame.min().min()
maxD = dualFrame.max().max()
#minP = minP if not isnan(minP) else maxD

for i, (l, tS, pS, dS) in enumerate(zip(descs, times, primals, duals)):
    rgb = tuple(rand(3))
    #lows, highs = zip(*[asRange(p-minP,maxD-d) for p,d in zip(pS,dS)])
    #semilogy(cumsum(tS),lows,label=l,c=rgb,marker='o')
    #semilogy(cumsum(tS),highs,c=rgb,marker='+')
    if pS.notnull().any():
        semilogy(cumsum(tS), pS - maxD, label=l+' (P)', c=rgb, marker='o')
    if dS.notnull().any():
        semilogy(cumsum(tS), minP - dS, label=l+' (D)', c=rgb, marker='+')
legend()
title('Weight finding @ (k=%d, n=%d, alpha=%.2f)' % (k, q, alpha))
xlabel('Time in seconds')
ylabel('Upper bound on sub optimality.')
show()
