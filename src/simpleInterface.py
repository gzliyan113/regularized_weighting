from numpy import array, inf, ones, zeros

from sklearn.svm import SVC
#from sklearn.metrics import zero_one_score

import alternatingAlgorithms as aa
import weightedModelTypes as wmt
from minL2PenalizedLossOverSimplex import penalizedMultipleWeightedLoss2, weightsForLosses, weightsCombinedForAM

#from exploreSvm import hinge_losses
#from svmutil import svm_train, svm_problem, svm_predict
#from svm import svm_parameter, SVC


def optimize(data, model_class, alpha, eta, model_parameters=None,
             dual_optimizer='coordinate-wise', dual_to_primal='partition',
             primal_optimizer='fista'):

    k = eta.shape[0]
    L = model_class.randomModelLosses(data, k, modelParameters=model_parameters)
    _, n = L.shape
    t = zeros(k)
    W = ones((k, n)) / n
    for i in range(10):
        W, t = weightsCombinedForAM(L, alpha, eta, ts0=t, W0=W) # solve for weights and t, with a relative duality gap or iteration complexity based stopping rules
        states = [model_class(data, w, model_parameters) for w in W]
        L = array([s.squaredLosses() for s in states])
    return states, t


def clustering(X, k, alpha, n_init=10):
    """
    Cluster the points in columns of X into k clusters.
    """
    minJointLoss = inf
    for i in xrange(n_init):
        #initialStates = aa.jointlyPenalizedInitialStates(X.T, wmt.ClusteringState, alpha, k, dualityGapGoal=1e-5)
        #finalStates = aa.learnJointlyPenalizedMultipleModels(initialStates, alpha, maxSteps=100, dualityGapGoal=1e-2)
        eta = ones(k) / k
        finalStates, t = optimize(X.T, wmt.ClusteringState, alpha, eta)
        centroids = [s.center for s in finalStates]
        W = array([s.weights for s in finalStates])
        L = array([s.squaredLosses() for s in finalStates])
        jointLoss = penalizedMultipleWeightedLoss2(L, W, alpha)
        if jointLoss < minJointLoss:
            minJointLoss = jointLoss
            highestWeights = W.argmax(0)
            zeroWeights = W.max(0) < 10 ** -9
            labels = highestWeights
            labels[zeroWeights] = -1
            bestCentroids = centroids

    return (bestCentroids, labels)


def mixtureGaussians(X, k, alpha, n_init=10):
    """
    Find a mixture of k Gaussians to explain points.
    """
    minJointLoss = inf
    for i in xrange(n_init):
        initialStates = aa.jointlyPenalizedInitialStates(X.T, wmt.ScalarGaussianState, alpha, k, dualityGapGoal=1e-5)
        finalStates = aa.learnJointlyPenalizedMultipleModels(initialStates, alpha, maxSteps=100, dualityGapGoal=1e-2)
        means = [s.mean for s in finalStates]
        W = array([s.weights for s in finalStates])
        L = array([s.squaredLosses() for s in finalStates])
        jointLoss = penalizedMultipleWeightedLoss2(L, W, alpha)
        if jointLoss < minJointLoss:
            minJointLoss = jointLoss
            highestWeights = W.argmax(0)
            zeroWeights = W.max(0) < 10 ** -9
            labels = highestWeights
            labels[zeroWeights] = -1
            bestMeans = means
            bestVariances = [s.variance for s in finalStates]

    return (bestMeans, bestVariances, labels)


def linearRegressionClustering(X, Y, k, alpha, regularizationStrength, n_init=10):
    minJointLoss = inf
    for i in xrange(n_init):
        initialStates = aa.jointlyPenalizedInitialStates((X.T, Y), wmt.MultiLinearRegressionState, alpha, k=k,
                                                         modelParameters={
                                                         'regularizationStrength': regularizationStrength})
        finalStates = aa.learnJointlyPenalizedMultipleModels(initialStates, alpha, maxSteps=10, dualityGapGoal=1e-2)
        linearModels = [s.r for s in finalStates]
        W = array([s.weights for s in finalStates])
        L = array([s.squaredLosses() for s in finalStates])
        jointLoss = penalizedMultipleWeightedLoss2(L, W, alpha)
        if jointLoss < minJointLoss:
            minJointLoss = jointLoss
            highestWeights = W.argmax(0)
            zeroWeights = W.max(0) < 10 ** -9
            associations = highestWeights
            associations[zeroWeights] = -1
            bestLinearModels = linearModels

    return (bestLinearModels, associations)


def sk_hinge_losses(clf, X, Y):
    decs = clf.decision_function(X)
    margin = Y * decs[:, 0]
    hlosses = 1 - margin
    hlosses[hlosses <= 0] = 0
    return hlosses


def sk_weightedCSVMrbf(Xain, Yain, Xst, Yst, alpha, C, gamma, n_init=10):
    clf = SVC(C=C, kernel='rbf', gamma=gamma)
    n_train_samples, n_dims = Xain.shape
    w = ones(n_train_samples) / n_train_samples
    #clf.fit(Xain, Yain)
    clf.fit(Xain, Yain, sample_weight=w * C + 0.0000000001)
    for _ in range(n_init):
        test_hlosses = sk_hinge_losses(clf, Xst, Yst)
        mistakeRatio = zero_one_score(Yst, clf.predict(Xst))
        print "mean loss on test data: %s %% correct on test data: %s" % (test_hlosses.mean(), mistakeRatio)
        train_hlosses = sk_hinge_losses(clf, Xain, Yain)
        w = weightsForLosses(train_hlosses, alpha)
        clf.fit(Xain, Yain, sample_weight=w * C + 0.0000000001)

    return clf, w


def hinge_losses(m, x, y):
    p_label, p_acc, p_val = svm_predict(y, x, m)
    margins = array([(ay * av) for (ay, [av]) in zip(y, p_val)])
    hlosses = 1 - margins
    hlosses[hlosses < 0] = 0
    return hlosses


def arrayToLibSvm(X):
    xlist = X.tolist()
    dim = len(xlist[0])
    return [dict(zip(range(dim), r)) for r in xlist]


def weightedCSVMrbf(Xain, Yain, Xst, Yst, alpha, C, gamma, n_init=10):
    xain = arrayToLibSvm(Xain)
    yain = list(Yain)
    xst = arrayToLibSvm(Xst)
    yst = list(Yst)
    n_train_samples, n_dims = Xain.shape

    param = svm_parameter('-c %f -g %f' % (C, gamma))

    W = list(C * ones(n_train_samples) / n_train_samples)
    for _ in range(n_init):
        prob = svm_problem(W, yain, xain)
        m = svm_train(prob, param)
        test_hlosses = hinge_losses(m, xst, yst)
        p_labels, _, _ = svm_predict(yst, xst, m)
        mistakeRatio = zero_one_score(Yst, array(p_labels))
        print "mean loss on test data: %s; accuracy on test data: %s" % (test_hlosses.mean(), mistakeRatio)
        train_hlosses = hinge_losses(m, xain, yain)
        W = list(C * weightsForLosses(train_hlosses, alpha))

    return m, W
