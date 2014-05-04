from numpy import array, allclose, zeros, newaxis, isnan, nonzero
from minL2PenalizedLossOverSimplex import weightsForMultipleLosses2TimedStreams
import utility
import pdb

def lloydsInitialStates(data, model, k=0, modelParameters=None):
    #print('Generating initial losses from random models.')
    while True:
        L = model.randomModelLosses(data, k, modelParameters)
        newW = lloydsIteration(L)
        if not isnan(newW).any():
            break

    return [model(data, w, modelParameters) for w in newW]

#@profile
def jointlyPenalizedInitialStates(data, model, alpha, k=0, modelParameters=None, dualityGapGoal=1e-5):
    #print('Generating initial losses from random models.')
    L = model.randomModelLosses(data, k, modelParameters)
    W, _ = weightsForMultipleLosses2TimedStreams(L, alpha, dualityGapGoal=dualityGapGoal)

    return [model(data, w, modelParameters) for w in W]


def lloydsIteration(L):
    bestModelPerPoint = L.argmin(0)
    newW = zeros(L.shape)
    for (b, i) in zip(bestModelPerPoint, utility.naturals()):
        newW[b, i - 1] = 1
    while 0 in newW.sum(1):
        badRows = nonzero(newW.sum(1) == 0)
        badRow = badRows[0]
        aGoodRow = bestModelPerPoint[badRow]
        newW[aGoodRow, badRow] = 0
        newW[badRow, badRow] = 1

    newW = newW / newW.sum(1)[newaxis].T
    return newW


def lloydsAlgorithm(initialStates, maxSteps=10):
    return multipleModelEvolution(initialStates,
                                  lambda L, W, s: (lloydsIteration(L), s),
                                  maxSteps=maxSteps)


def learnJointlyPenalizedMultipleModels(initialStates, alpha, maxSteps=10, dualityGapGoal=1e-5):
    return multipleModelEvolution(initialStates,
                                  lambda L, W, l2: weightsForMultipleLosses2TimedStreams(L, alpha, W=W, l2=l2,
                                                                                         dualityGapGoal=dualityGapGoal),
                                  maxSteps=maxSteps)


def weightsClose(w1, w2):
    k, q = w1.shape
    return allclose(w1, w2, rtol=10 ** -2, atol=((1 / q) * 10 ** -2))


def multipleModelEvolution(initialStates, weightMapping, maxSteps=10):
    print "starting mme"
    es = initialStates[0]
    W = array([s.weights for s in initialStates])
    sideInformation = None
    # bound computation time
    for _ in range(maxSteps):
        print('Fitting new models')
        currStates = [es.nextState(w) for w in W]
        L = array([currState.squaredLosses() for currState in currStates])
        newW, sideInformation = weightMapping(L, W, sideInformation)
        if weightsClose(newW, W):
            break
        W = newW
    print "done mme"
    return currStates
