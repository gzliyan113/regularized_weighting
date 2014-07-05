from numpy import allclose
from numpy.random import rand, randn
from lambdaAndWinnersCache import LambdaAndWinnersTree
import pdb


def test_family():
    k = 13
    n = 10
    L = rand(k, n)
    t = randn(k)

    c = LambdaAndWinnersTree(L, t)
    for j in range(k):
        p = c.parent_of_model(j)
        assert p == c.parent_of_model(c.model_paired_with(j))

    rows, _ = c.cache.shape
    for h in range(rows):
        p = c.parent_of(h)
        assert p == c.parent_of(c.paired_with(h))


def test_basic_scenario():
    k = 131
    n = 10
    L = rand(k, n)
    t = randn(k) / 5

    c = LambdaAndWinnersTree(L, t)
    #pdb.set_trace()
    lam1, win1 = c.lambda_and_winners()
    print lam1
    print (t - L.T).T
    assert allclose(lam1, (t - L.T).T.max(0))
    for j in range(k):
        all_but_j = [j_ for j_ in range(k) if j_ != j]
        lam2, win2 = c.lambda_and_winners_all_but(j)
        print "lam2: " + str(lam2)
        print "manual: " + str((t[all_but_j] - L[all_but_j, :].T).T.max(0))
        assert allclose((t[all_but_j] - L[all_but_j, :].T).T.max(0), lam2)

