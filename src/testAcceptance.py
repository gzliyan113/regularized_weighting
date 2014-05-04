from nose.plugins.attrib import attr

from numpy.testing import assert_array_less
from numpy import array, hstack
from numpy.random import rand
from hopkinsTesting import runOnManyExtraNoise, findBalancedSets, manyNames, rest
import alternatingAlgorithms
import utility
import weightedModelTypes


@attr('slow')
def testBeatLloydsOnRobustMeasureHopkins():
    "Our method vs Lloyd at high noise on two Hopkins datasets"
    imbalanceRatios, names = findBalancedSets(manyNames + rest)
    numberToRun = 2
    imbalanceRatios = imbalanceRatios[0:numberToRun]
    names = names[0:numberToRun]

    noiseLevel = 0.2
    multipleSummary = runOnManyExtraNoise(names,
                                          lambda sSet: utility.medianAbsLoss(sSet),
                                          iters=10, noiseLevel=noiseLevel)

    lloydsMedianAbsLoss, lloydsR, ourMedianAbsLoss, ourR, toplineQ, toplineR = zip(
        *[(allLloydQualities.min(),
           lloydRunMissRates[allLloydQualities.argmin()],
           allQualities.min(),
           missRates[allQualities.argmin()],
           toplineQuality, toplineMissRate)

          for (name, size, allQualities, allLloydQualities, toplineQuality,
               missRates, lloydRunMissRates, toplineMissRate)
          in multipleSummary])
    assert_array_less(array(ourMedianAbsLoss), array(lloydsMedianAbsLoss))


@attr('slow')
def testBeatLloydsOnRobustMeasureQuadratic():
    "Our method vs Lloyds on quadratic regression, should have more eps correct answers."
    coeffs = [array([-0.04780819, -0.3844407, -1.73699185]),
              array([-1.44054051, -0.63069834, -0.9437272])]
    q1 = 40
    x1 = rand(1, 40)
    x2 = rand(1, 40)
    y1 = weightedModelTypes.quadraticKernelMapping(x1.T).dot(coeffs[0]) + 0.05 / rand(q1)
    y2 = weightedModelTypes.quadraticKernelMapping(x2.T).dot(coeffs[1]) + 0.05 / rand(q1)

    # some convenience
    x = hstack((x1, x2))
    y = hstack((y1, y2))
    y = y[x.argsort()].flatten()
    x.sort()
    _, q = x.shape

    alpha = 100

    ourStateSets = []
    lloydsStateSets = []
    for _ in range(20):
        ourInitStates = alternatingAlgorithms.jointlyPenalizedInitialStates((x, y),
                                                                            weightedModelTypes.MultiQuadraticRegressionState,
                                                                            alpha,
                                                                            k=2)

        ourStateSets.append(alternatingAlgorithms.learnJointlyPenalizedMultipleModels(ourInitStates,
                                                                                      alpha,
                                                                                      maxSteps=10))

        lloydsInitStates = alternatingAlgorithms.lloydsInitialStates((x, y),
                                                                     weightedModelTypes.MultiQuadraticRegressionState,
                                                                     k=2)

        lloydsStateSets.append(alternatingAlgorithms.lloydsAlgorithm(lloydsInitStates, maxSteps=10))

    ourBestSSet = ourStateSets[array([utility.medianAbsLoss(sSet)
                                      for sSet
                                      in ourStateSets]).argmin()]

    lloydsBestSSet = lloydsStateSets[array([utility.medianAbsLoss(sSet)
                                            for sSet
                                            in lloydsStateSets]).argmin()]

    def epsCorrect(s, eps):
        return ((s.squaredLosses() < eps) * 1.).sum()

    eps = 0.25
    assert (sum([epsCorrect(s, eps) for s in ourBestSSet]) >=
            sum([epsCorrect(s, eps) for s in lloydsBestSSet]))
