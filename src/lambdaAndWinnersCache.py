from numpy import (ones, vstack, zeros, ceil, log2, infty, where, maximum)
from optimized_rw_core import update_win_lam
import pdb


class LambdaAndWinnersTree:
    """ A cache for maximizers and maximum values of t-L.

     The cache is a complete tree sufficient to have a leaf for every pair of
     models. If k is not a power two, missing models are treated as having
     infinite loss.
     leaves at 1-2, etc. """
    def __init__(self, L, t):
        self.L = L
        self.t = t.copy()
        self.k, self.n = L.shape
        self.levels = int(ceil(log2(self.k)))
        cache_rows = (2 ** self.levels) - 1
        self.cache = -infty * ones((cache_rows, self.n))
        self.winners = -ones((cache_rows, self.n))
        self.initial = -infty * ones(self.n)
        for j in range(self.k):
            self.update_t_at(j, t[j])

    def lambda_and_winners(self):
        return self.cache[0, :], self.winners[0, :]

    #@profile
    def lambda_and_winners_all_but(self, j):
        # begin with lambda and winners of paired_model
        curr_lam, curr_win = self.lambda_winners_model_paired_with(j)

        curr_cache = self.parent_of_model(j)

        # Augment with cached information from higher levels.
        while curr_cache > 0:
            other_cache = self.paired_with(curr_cache)
            other_win = self.winners[other_cache, :]
            other_lam = self.cache[other_cache, :]

            # update_win_lam(curr_lam, curr_win, other_lam, other_win)
            # Order of next two updates matters!!
            curr_win = where(other_lam > curr_lam, other_win, curr_win)
            curr_lam = maximum(other_lam, curr_lam)

            curr_cache = self.parent_of(curr_cache)

        return curr_lam, curr_win

    def parent_of_model(self, j):
        return self.parent_of(j+(2**self.levels - 1))

    #@profile
    def update_t_at(self, j, tj):
        self.t[j] = tj
        other_lam, other_win = self.lambda_winners_model_paired_with(j)
        son_lam = tj - self.L[j, :]
        son_win = j * ones(self.n)

        curr_cache = self.parent_of_model(j)
        while curr_cache > 0:
            # Order of next two updates matters!!
            son_win = where(other_lam > son_lam, other_win, son_win)
            son_lam = maximum(other_lam, son_lam)

            self.cache[curr_cache, :] = son_lam
            self.winners[curr_cache, :] = son_win

            other_lam = self.cache[self.paired_with(curr_cache), :]
            other_win = self.winners[self.paired_with(curr_cache), :]

            curr_cache = self.parent_of(curr_cache)

        # update root
        son_win = where(other_lam > son_lam, other_win, son_win)
        son_lam = maximum(other_lam, son_lam)
        self.winners[curr_cache, :] = son_win
        self.cache[curr_cache, :] = son_lam

    @staticmethod
    def parent_of(j):
        return (j - 1) / 2

    @staticmethod
    def paired_with(j):
        # add one if odd, deduct one if even.
        return j + (((j % 2) * 2) - 1)

    @staticmethod
    def model_paired_with(j):
        # add one if even, deduct one if odd.
        return j - (((j % 2) * 2) - 1)

    def lambda_winners_model_paired_with(self, j):
        oj = self.model_paired_with(j)
        if oj < self.k:
            return self.t[oj] - self.L[oj, :], oj*ones(self.n)
        else:
            return self.initial, -1*ones(self.n)

    def update_L_at(self, j, l):
        self.L[j, :] = l
        self.update_t_at(j, self.t[j])