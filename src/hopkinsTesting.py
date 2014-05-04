# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:58:33 2012

@author: danielv
"""
import time
from itertools import count
from numpy import (array, sqrt, arange, linspace, unique, median, ceil,
                   flatnonzero)
from numpy.random import randn, rand

from matplotlib.pyplot import (plot, show, subplot, title, semilogy,
                               figure, legend)

from hopkinsUtilities import loadSet, hopkinsDatasetDimensions
from utility import meanMinSquaredLoss, segmentationError
import alternatingAlgorithms
import weightedModelTypes
import utility

#name = 'three-cars'
#name = '1RT2TC'


def subsetsAsUniformDistributions(S):
    dists = []
    for s in unique(S):
        subset = (S == s) * 1.0
        c = sum(subset)
        dists.append(subset / c)
    return dists


def runAlgorithm(data, d, k):
    W = None

    alphas = 10.0 ** linspace(-4, 1, 6)
    #alphas = 10.0 ** array([-2,-1,0,-2,-3,-2])
    expNum = len(alphas)
    allStates = []
    for (alpha, i) in zip(alphas, count(1)):
        states = jointlyPenalizedMultiplePCA(data, d, alpha, W, k=k)
        allStates.append(states)
        weightVectors = [s.weights for s in states]
        W = array(weightVectors)

    figure()
    for (states, alpha, i) in zip(allStates, alphas, count(1)):
        subplot(expNum, 1, i)
        weightVectors = [s.weights for s in states]
        W = array(weightVectors)
        plot(W.T)
        title(r'Run with $\alpha $ = %f' % alpha)

    figure()
    for (states, alpha, i) in zip(allStates, alphas, count(1)):
        subplot(expNum, 1, i)
        lossVectors = [s.squaredLosses() for s in states]
        L = array(lossVectors)
        semilogy(L.T)
        title(r'Run with $\alpha $ = %f' % alpha)

    show()


def preprocessHopkins(name):
    (rawData, S) = loadSet('/home/danielv/Research/Datasets/Hopkins155',
                           name)
    if S.shape[0] == 0:
        print('Bad name: %s' % name)
        return (0, 0, 0, 0, 0, 0)

    dims = hopkinsDatasetDimensions(rawData, S)
    d = int(ceil(median(dims)))
    k = len(dims)
    sufficientDimension = sum(dims)
    lowData = utility.PCA(rawData, sufficientDimension)
    lowNormedData = lowData / sqrt(sum(lowData ** 2, 0))
    data = lowNormedData
    _, q = lowNormedData.shape
    return (dims, S, data, d, k, q)


def findBadNames(names):
    for name in names:
        (dims, S, data, d, k, q) = preprocessHopkins(name)
        if dims == 0:
            print name


def findBalancedSets(names):
    summary = []
    for name in names:
        (dims, S, data, d, k, q) = preprocessHopkins(name)
        if dims == 0:
            print(name + 'is bad')
            continue
        classSizes = [flatnonzero(S == v).size for v in unique(S)]
        summary.append((float(min(classSizes)) / max(classSizes), name))
    summary.sort(reverse=True)
    return zip(*summary)


def testLloydsOnMany(names):
    summary = []
    for name in names:
        (dims, S, data, d, k, q) = preprocessHopkins(name)
        if dims == 0:
            print('Directory named: %s has no data.' % name)
            continue
        print('Dataset %s, k=%d, dims = %s, using d=%d' % (name, k, dims, d))

        toplineStateSet = [weightedModelTypes.MultiPCAState(data[:, :], w, d)
                           for w
                           in subsetsAsUniformDistributions(S)]
        toplineMeanError = meanMinSquaredLoss(toplineStateSet)

        allRuns = [runLloydsNonString(data, d, k) for _ in arange(10)]
        allStateSets = [sSet for run in allRuns for sSet in run]
        allMSEs = array([meanMinSquaredLoss(sSet) for sSet in allStateSets])
        bestRun = allMSEs.argmin()
        _, bestRunMissRate = segmentationError(allStateSets[bestRun], S)
        print('best/topline error: %f' % (toplineMeanError / allMSEs.min()))
        print('Misclassification rate: %f' % bestRunMissRate)
        summary.append((name, allMSEs, S.size,
                        bestRunMissRate, toplineMeanError))
    return summary


def testOnMany(names, internalQuality, iters=10):
    summary = []
    for name in names:
        (dims, S, data, d, k, q) = preprocessHopkins(name)
        if dims == 0:
            print('Directory named: %s has no data.' % name)
            continue
        summary.append(testDataSet(name, S, dims, data,
                                   k, d, internalQuality, iters=iters))
    return summary


def runOnManyExtraNoise(names, internalQuality, iters=10, noiseLevel=0):
    summary = []
    totalCount = len(names)
    for (i, name) in zip(count(1), names):
        (dims, S, data, d, k, q) = preprocessHopkins(name)
        if dims == 0:
            print('Directory named: %s has no data.' % name)
            continue
        summary.append(testDataSet(name, S, dims, data,
                                   k, d, internalQuality, iters=iters,
                                   noiseLevel=noiseLevel))
        print('Finished %d out of %d' % (i, totalCount))
        print(time.localtime())
    return summary


def addNoise(data, level):
    D, q = data.shape
    raw = data + randn(D, q) * level / rand(q)
    return raw / raw.sum(0)


def testDataSet(name, S, dims, data, k, d, internalQuality, iters=10, noiseLevel=0):
    print('Dataset %s, k=%d, dims = %s, using d=%d' % (name, k, dims, d))
    data = addNoise(data, noiseLevel)
    allRuns = [runString(data, d, k) for _ in arange(iters)]
    allStateSets = [sSet for run in allRuns for sSet in run]
    allQualities = array([internalQuality(sSet) for sSet in allStateSets])
    bestRun = allQualities.argmin()
    _, missRates = zip(*[segmentationError(sSet, S) for sSet in allStateSets])
    missRates = array(missRates)

    allLloydRuns = [runLloydsNonString(data, d, k) for _ in arange(iters)]
    allLloydStateSets = [sSet for run in allLloydRuns for sSet in run]
    allLloydQualities = array([internalQuality(sSet)
                               for sSet in allLloydStateSets])
    bestLloydRun = allLloydQualities.argmin()
    _, lloydRunMissRates = zip(*[segmentationError(sSet, S)
                                 for sSet in allLloydStateSets])
    lloydRunMissRates = array(lloydRunMissRates)

    toplineStateSet = [weightedModelTypes.MultiPCAState(data[:, :], w, {'d': d})
                       for w
                       in subsetsAsUniformDistributions(S)]
    toplineQuality = internalQuality(toplineStateSet)
    _, toplineMissRate = segmentationError(toplineStateSet, S)

    print('our:Lloyds:topline quality: %f:%f:%f'
          % (allQualities.min(),
             allLloydQualities.min(),
             toplineQuality))
    print('Misclassification rate: (our:Lloyds:topline)  %f:%f:%f' %
          (missRates[bestRun], lloydRunMissRates[bestLloydRun], toplineMissRate))
    return (
    name, S.size, allQualities, allLloydQualities, toplineQuality, missRates, lloydRunMissRates, toplineMissRate)


def toplineOnMany(names):
    summary = []
    for name in names:
        (dims, S, data, d, k, q) = preprocessHopkins(name)
        if dims == 0:
            print('Directory named: %s has no data.' % name)
            continue
        print('Dataset %s, k=%d, dims = %s, using d=%d' % (name, k, dims, d))

        toplineStateSet = [weightedModelTypes.MultiPCAState(data[:, :], w, d)
                           for w
                           in subsetsAsUniformDistributions(S)]
        _, missClassificationRate = segmentationError(toplineStateSet, S)
        toplineMeanError = meanMinSquaredLoss(toplineStateSet)
        summary.append((name, S.size,
                        missClassificationRate, toplineMeanError))
    return summary


def runString(data, d, k, lowAlpha=10 ** -2, highAlpha=10 ** 1):
    #alphas = 10.0 ** linspace(lowAlpha, highAlpha, 6)
    #initStates = jointlyPenalizedMultiplePCAInitialStates(data, d, alphas[0], k=k)
    #allStates = iterates(initStates, alphas, lambda s, alpha:
    #                       learnJointlyPenalizedMultipleModels(s, alpha))
    #return allStates

    # alpha between 10 and 1000:
    #alpha = 10 ** (rand() * 3 - 2)
    dataDim, n = data.shape
    # alpha between 0.1n and 10n, where n = # data points
    alpha = n * (10 ** (rand() * 2 - 1))
    print "alpha %s" % alpha
    return [alternatingAlgorithms.learnJointlyPenalizedMultipleModels(
        alternatingAlgorithms.jointlyPenalizedInitialStates(data, weightedModelTypes.MultiPCAState, alpha, k, {'d': d}),
        alpha)]


def runLloydsNonString(data, d, k):
    initStates = alternatingAlgorithms.lloydsInitialStates(data, weightedModelTypes.MultiPCAState, k=k,
                                                           modelParameters={'d': d})
    return [alternatingAlgorithms.lloydsAlgorithm(initStates)]


def showMissclassificationRates(summary):
    (mmsers, missRates, sizes, setNumber) = zip(
        *[[MSEs.min() / t, missRate, numPoints, i]
          for ((n, MSEs, numPoints, missRate, t), i)
          in zip(summary, count(0))])
    figure()
    missRates = array(missRates)
    sizes = array(sizes)

    missRatesOrder = missRates.argsort()
    semilogy(missRates[missRatesOrder], marker='.')
    semilogy(sizes[missRatesOrder], marker='.')
    legend(['missclassification rate, if positive', 'points in dataset'])
    title('missclassification rates and dataset sizes.')
    print('Mean missclassification rate: %f' %
          (sum(missRates * sizes) / sum(sizes)))
    print('Median missclassification rate: %f' %
          median(missRates))
    show()


def showRepresentationErrors(summary):
    (mmsers, missRates, sizes, setNumber) = zip(
        *[[MSEs.min() / t, missRate, numPoints, i]
          for ((n, MSEs, numPoints, missRate, t), i)
          in zip(summary, count(0))])
    figure()
    mmsers.sort()
    semilogy(mmsers, marker='.')
    #semilogy([t for (n, MSEs, b, t) in summary])
    legend(['best of 60 MSE / oracle MSE', 'baseline'])  # topline',
    title('Hopkins 155')
    show()

# testOnMany(names = ['dancing', 'three-cars', '1RT2TC'])

fewNames = ['2T3RTCR_g23', 'articulated_g12', '2RT3RCT_B_g23']
manyNames = ['2T3RTCR_g23', 'articulated_g12', '2RT3RCT_B_g23',
             '2RT3RTCRT_g23', '1RT2RCRT_g12', '2T3RCRTP_g12', '1R2RCT_A_g23',
             'three-cars_g13', '1RT2TC_g13', 'cars7', '1R2RCT_B_g23',
             '1R2TCR_g13', 'arm', '1RT2RCRT_g23', 'cars10_g12', '2RT3RTCRT_g13',
             'articulated', '1RT2RCR_g12', 'cars10_g23', '1R2RCT_B',
             '1R2RC_g23']
# knownBad = ['1R2RC']
rest = ['2R3RTC_g23', 'cars6', '1RT2TC', '1R2RCT_A', '2R3RTCRT_g13',
        '1RT2TC_g23', '1R2RCT_B_g12', '2RT3RCR_g13', '2T3RCTP_g13',
        '1R2TCRT_g12', '2T3RCR', '1R2RCT_A_g13', '1R2RCT_A_g12',
        'cars2B_g12', 'kanatani2', '1RT2TCRT_B_g13', '2T3RCRTP_g13',
        '2RT3RC_g12', 'kanatani3', '2R3RTC', '1R2RCR_g13', '1R2RC_g12',
        '2RT3RC', '2T3RCTP', 'cars2_06_g23', '2RT3RC_g23', 'cars9_g13',
        '1RT2RCRT_g13', '2R3RTCRT_g23', '2R3RTC_g13', 'cars3_g12', 'dancing',
        '2T3RCRT_g23', '1RT2RTCRT_A_g13', 'three-cars', 'cars2_07',
        'cars2B', 'cars5_g23', 'cars9_g12', '2R3RTCRT', '2RT3RCT_A_g13',
        '2RT3RCT_A', '2T3RCR_g23', '2RT3RCT_A_g23', 'cars9',
        '1RT2TCRT_A_g12', 'three-cars_g12', 'cars4', '1R2TCR_g12',
        'cars3_g13', 'cars2B_g23', 'truck2', '2RT3RCT_B_g12',
        'two_cranes_g23', '2T3RCR_g12', '1RT2TCRT_B_g23', '1R2TCR_g23',
        'cars5_g12', 'two_cranes', '1RT2RTCRT_B_g12', '2RT3RCT_B',
        '2T3RCR_g13', 'cars1', '1R2RCR', '1R2RCR_g12', '1R2RCR_g23',
        'kanatani1', '1R2TCRT_g13', '1RT2RCRT', '2T3RTCR', 'truck1',
        '2R3RTC_g12', '2RT3RCR', '2T3RCTP_g12', '1R2RCT_B_g13',
        'two_cranes_g13', '1RT2RTCRT_B', '1RT2RCR_g23', '1RT2TCRT_A_g23',
        '1RT2RTCRT_A_g23', 'cars10_g13', '2T3RCRT_g12', '1R2RC_g13',
        '1RT2TCRT_B', '2RT3RCT_A_g12', 'cars2B_g13', 'cars5', '2RT3RCT_B_g13',
        'articulated_g13', '2RT3RTCRT', 'people2', '2RT3RCR_g23', '2RT3RC_g13',
        'cars3_g23', '1R2TCR', 'head', '1R2TCRT', 'three-cars_g23',
        'two_cranes_g12', '2T3RTCR_g13', '1RT2RTCRT_A', '1RT2TCRT_A',
        '2T3RTCR_g12', 'cars5_g13', 'articulated_g23', '2R3RTCRT_g12',
        '1RT2TCRT_A_g13', 'cars2_06_g13', 'people1', 'cars9_g23',
        '1R2TCRT_g23', 'cars2_07_g12', '1RT2RCR_g13', '1RT2RTCRT_B_g13',
        'cars3', '2T3RCTP_g23', '2T3RCRT_g13', '1RT2TCRT_B_g12',
        'cars2_07_g23', '2RT3RTCRT_g12', 'cars2_06', '2T3RCRTP', 'cars8',
        '1RT2TC_g12', 'cars2_07_g13', 'cars2', 'cars10', '1RT2RTCRT_A_g12',
        '1RT2RTCRT_B_g23', '2T3RCRT', '2T3RCRTP_g23', 'cars2_06_g12',
        '1RT2RCR', '2RT3RCR_g12']
#summary = testOnMany(manyNames+rest)
#summary = testOnMany(fewNames)
#file('summaryFewBackup.txt', 'w').write(repr(summary))

#(brs, mmsers, s) = zip(*[[b/t,mse.min()/t, i]
#               for ((n, MSEs, b, t), i) in zip(summary, count(0))
#               for mse in MSEs.flatten()])

#name = '1RT2TC'
#(dims, S, data, d, k, q) = preprocessHopkins(name)
#runAlgorithm(data, d, k)
