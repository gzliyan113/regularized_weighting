from numpy import logical_not, sqrt, ones, logspace, linspace, inf, newaxis, log2, median, argmin, eye, diag, dot
from numpy.linalg import norm, svd
from math import floor, log10, ceil
from numpy.numarray import arange
import simpleInterface as si
import numpy as np
from numpy.random import randn
from numpy import array, zeros, allclose
from minL2PenalizedLossOverSimplex import weightsForLosses, penalizedMultipleWeightedLoss2
from robust_pca import shrinkage_pca
from weightedModelTypes import MultiLinearRegressionState, ClusteringState, MultiPCAState
import pdb
from utility import squaredColNorm, nonNegativePart
from random import sample
import matplotlib.pyplot as plt


def create_regression_task(d, n, noise_strength, alt_source_strength, noisy_proportion=0.1):
    X = randn(d, n) / np.sqrt(d)
    rgt = np.zeros(d)
    rgt[0] = 1
    rns = randn(d) / np.sqrt(d)
    #print("rns = %s" % rns)
    y = rgt.dot(X) + randn(n) / 10
    first_outlier = int(ceil((1 - noisy_proportion) * n))
    #print("first_outlier = %s" % first_outlier)
    outlier_mods = rns.dot(X[:, first_outlier:]) * alt_source_strength + \
                   randn(n - first_outlier) * noise_strength
    #print("outlier_mods %s" % outlier_mods)
    y[first_outlier:] += outlier_mods
    return X, y, rgt


def linear_regression_as_function_of_beta():
    n_samples = 1000
    X, y, rgt = create_regression_task(10, n_samples, 100, 0)
    n_betas = 10
    u = ones(n_samples) / n_samples
    dists_from_uniform = zeros(n_betas)
    nnz_counts = zeros(n_betas)
    inlier_mses = zeros(n_betas)
    for i, beta in enumerate(logspace(0.01, 100, n_betas)):
        s = weighted_regression(X.T, y, beta)
        dists_from_uniform[i] = norm(s.weights - u)
        nnz_counts[i] = np.count_nonzero(s.weights)
        inlier_mses[i] = s.squaredLosses()[:900].mean()
    return dists_from_uniform, nnz_counts, inlier_mses


def experiment_matrix_results(make_data, row_param_values, col_param_values, methods, summarize):
    n_cols = col_param_values.shape[0]
    n_rows = row_param_values.shape[0]
    result_matrices = [ones((n_rows, n_cols)) * inf for m in methods]
    for ip, noise_prop in enumerate(col_param_values):
        for io, offset in enumerate(row_param_values):
            print("."),
            data, gt = make_data(offset, noise_prop)
            for i, (m, _) in enumerate(methods):
                estimate = m(data)
                result_matrices[i][io, ip] = summarize(estimate, gt)
        print(".")
    method_names = [n for (m, n) in methods]
    return result_matrices, method_names


def annoying_pca_data(dimension, big_dims, total_samples, noisy_proportion, noise_level, disturbance_level):
    # Base data
    noise_scale = dimension ** -0.5
    scales = ones(dimension) * noise_scale * noise_level
    scales[:big_dims] = big_dims ** -0.5
    X = randn(total_samples, dimension).dot(diag(scales))

    # Adding disturbance
    noisy_samples = int(floor(total_samples * noisy_proportion))
    A = zeros((dimension, dimension))
    A[-1, 0] = 1
    X[:noisy_samples, :] += A.dot(X[:noisy_samples, :].T).T * disturbance_level

    return X, eye(dimension)


def angleCosinesBetweenBases(baseForFirst, baseForSecond):
    prod = dot(baseForFirst.transpose(), baseForSecond)
    _, s, _ = svd(prod, full_matrices=False)
    return s


def pca_experiment_matrix_results(big_dim, d, samples, noisy_proportions, disturbance_levels):
    def make_task(noise_prop, dist_lvl):
        X, gt_basis = annoying_pca_data(d, big_dim, samples, noise_prop, 0.1, dist_lvl)
        return X, gt_basis

    def score(basis, gt_basis):
        return norm(1 - angleCosinesBetweenBases(basis[:, :big_dim], gt_basis[:, :big_dim]))

    def huber1(X):
        return shrinkage_pca(X, big_dim, 0.1).U

    def huber1_5(X):
        return shrinkage_pca(X, big_dim, 0.0333).U

    def huber2(X):
        return shrinkage_pca(X, big_dim, 0.01).U

    def weighted(X):
        beta = 1
        return weighted_pca(X.T, big_dim).U

    def standard(X):
        samples, _ = X.shape
        u = ones(samples) / samples
        return (MultiPCAState(X.T, u, modelParameters={'d': big_dim})).U

    return experiment_matrix_results(make_task, noisy_proportions, disturbance_levels,
                                     [(huber1, r"Huber-loss with $\lambda = $ 0.1."),
                                      (huber1_5, r"Huber-loss with $\lambda = $ 0.0333."),
                                      (huber2, r"Huber-loss with $\lambda = $ 0.01."),
                                      (weighted, "RW, automatically set beta."),
                                      (standard, "Regular PCA.")],
                                     score)


def display_result_matrices(matrices, method_names, row_vals, col_vals, ylabel, xlabel):
    high = max([m.max() for m in matrices])
    low = min([m.min() for m in matrices])
    n_methods = len(method_names)
    for i, (m, n) in enumerate(zip(matrices, method_names)):
        plt.subplot(1, n_methods, i+1)
        plt.imshow(m, cmap=plt.cm.gray_r, vmin=low, vmax=high, interpolation='nearest')
        plt.title(n)
        plt.yticks(arange(row_vals.shape[0]), ["%.2f" % rv for rv in row_vals])
        plt.xticks(arange(col_vals.shape[0]), ["%.2f" % cv for cv in col_vals], rotation=70)
        plt.colorbar()
        plt.xlabel(xlabel)
        if i == 0:
            plt.ylabel(ylabel)


def linear_regression_experiment_matrix_results(d, n, noise_strengths, alt_src_strengths):
    def make_task(noise_str, alt_src_str):
        X, y, rgt = create_regression_task(d, n, noise_str, alt_src_str)
        return (X, y), rgt

    def score(r, rgt):
        return norm(r - rgt)

    def huber(data):
        X, y = data
        return huber_regression(X.T, y, 0.1)

    def weighted(data):
        X, y = data
        return weighted_regression2(X.T, y).r

    return experiment_matrix_results(make_task, noise_strengths, alt_src_strengths, [huber, weighted], score)


def linear_regression_experiment_matrix_results2(d, n, noisy_props, alt_src_strengths):
    def make_task(noisy_prop, alt_src_str):
        X, y, rgt = create_regression_task(d, n, 0.1, alt_src_str, noisy_proportion=noisy_prop)
        return (X, y), rgt

    def score(r, rgt):
        return norm(r - rgt)

    def huber(data):
        X, y = data
        return huber_regression(X.T, y, 0.1)

    def weighted(data):
        X, y = data
        return weighted_regression2(X.T, y).r

    def regular(data):
        X, y = data
        n, d = X.T.shape
        u = ones(n) / n
        s = MultiLinearRegressionState((X, y), u, {'regularizationStrength': 0})
        return s.r

    return experiment_matrix_results(make_task, noisy_props, alt_src_strengths,
                                     [(huber, "Min. Huber Loss."),
                                     (weighted, "RW loss."),
                                     (regular, "std. regression.")],
                                     score)


def huber_regression(point_rows, y, eps):
    n, d = point_rows.shape
    u = ones(n) / n
    s = MultiLinearRegressionState((point_rows.T, y), u, {'regularizationStrength': 0})
    r = s.r
    old_r = zeros(d)
    while not allclose(old_r, r):
        old_r = r
        predicted_y = r.dot(point_rows.T)
        correction = soft_shrink_rows((predicted_y - y)[:, np.newaxis], eps).flatten()
        f_y = correction + y
        s = MultiLinearRegressionState((point_rows.T, f_y), u, modelParameters={'regularizationStrength': 0})
        r = s.r
    return r


def weighted_regression(point_rows, y, beta):
    samples, dim = point_rows.shape
    alpha = beta * samples
    w = ones(samples) / samples
    w_old = zeros(samples)
    s = None
    while not allclose(w, w_old):
        w_old = w
        s = MultiLinearRegressionState((point_rows.T, y), w, {'regularizationStrength': 0})
        w = weightsForLosses(s.squaredLosses(), alpha)
    return s


def weighted_regression2(point_rows, y):
    s, beta = weighted_model_find_beta2((point_rows.T, y), MultiLinearRegressionState, {'regularizationStrength': 0})
    #print("beta is %s" % beta)
    n = s.weights.shape[0]
    #print("norm(w-1/n) * sqrt(n) = %s" % (norm(s.weights - (ones(n) / n)) * sqrt(n)))
    return s


def location_estimation(data, beta):
    s = weighted_modeling(data, ClusteringState, None, beta)
    return s.center


def location_estimation2(data):
    s, beta = weighted_model_find_beta2(data, ClusteringState, None)
    #print("beta is %s" % beta)
    #n = s.weights.shape[0]
    #print("norm(w-1/n) * sqrt(n) = %s" % (norm(s.weights - (ones(n) / n)) * sqrt(n)))
    return s.center


def weighted_pca(data, d):
    s, beta = weighted_model_find_beta2(data, MultiPCAState, {'d': d})
    return s


def weighted_model_find_beta(data, model_class, model_parameters):
    beta = 1.
    s = weighted_modeling(data, model_class, model_parameters, beta)
    n = s.weights.shape[0]

    def w_distance(s):
        return sqrt((norm((n * s.weights) - 1) ** 2) / n)

    max_distance = 1.1
    #plt.plot(s.weights, label='wdis = {0:f}, beta = {1:f}, werr = {2:f}'.format(w_distance(s), beta,
    #                                                                            s.weights.dot(s.squaredLosses())))
    if w_distance(s) > max_distance:
        print("started at w_dis = %s, doubling beta" % w_distance(s))
        while w_distance(s) > max_distance:
            beta *= 2
            s = weighted_modeling(data, model_class, model_parameters, beta)
            #print("w_dis = %s, at beta = %f" % (w_distance(s), beta))
        print("finished at w_dis = %s" % w_distance(s))
    else:
        print("started at w_dis = %s, halving beta" % w_distance(s))
        while w_distance(s) <= max_distance:
            beta /= 2
            s = weighted_modeling(data, model_class, model_parameters, beta)
            print("w_dis = %s, at beta = %f, werr: %f" % (w_distance(s), beta, s.weights.dot(s.squaredLosses())))
            #plt.plot(s.weights, label='wdis = {0:f}, beta = {1:f}, werr = {2:f}'.format(w_distance(s), beta,
            #                                                                            s.weights.dot(s.squaredLosses())))

        print("finished at w_dis = %s, doubling once." % w_distance(s))
        beta *= 2
        s = weighted_modeling(data, model_class, model_parameters, beta)
        print("final w_dis = %s, doubling once." % w_distance(s))
        #plt.legend(loc='best')
    #plt.show()
    return s, beta


def weighted_model_find_beta2(data, model_class, model_parameters):
    beta = 1e6
    s = weighted_modeling(data, model_class, model_parameters, beta)
    n = s.weights.shape[0]
    u = ones(n) / n
    uni_s = model_class(data, u, model_parameters)

    def w_distance(s):
        return sqrt((norm((n * s.weights) - 1) ** 2) / n)

    beta = uni_s.weights.dot(uni_s.squaredLosses())
    s = weighted_modeling(data, model_class, model_parameters, beta)
    '''max_distance = 1.1
    plt.plot(s.weights, label='wdis = {0:f}, beta = {1:f}, werr = {2:f}'.format(w_distance(s), beta,
                                                                                s.weights.dot(s.squaredLosses())))

    median_error_at_beta = dict()
    while w_distance(s) <= max_distance:
        beta /= 2
        s = weighted_modeling(data, model_class, model_parameters, beta)
        median_error_at_beta[beta] = median(s.squaredLosses())
        print("w_dis = %s, at beta = %f, werr: %f" % (w_distance(s), beta, s.weights.dot(s.squaredLosses())))
        plt.plot(s.weights, label='wdis = {0:f}, beta = {1:f}, werr = {2:f}'.format(w_distance(s), beta,
                                                                                    s.weights.dot(s.squaredLosses())))

    betas_col_err_col = array(list(median_error_at_beta.iteritems()))
    best_beta_loc = argmin(betas_col_err_col[:, 1])
    best_beta = betas_col_err_col[best_beta_loc, 0]
    best_s = weighted_modeling(data, model_class, model_parameters, best_beta)
    plt.legend(loc='best')
    #plt.show()
    plt.scatter(log2(betas_col_err_col[:, 0]), betas_col_err_col[:, 1])
    #plt.show()
    print(median_error_at_beta)
    return best_s, best_beta'''
    return s, beta


def weighted_modeling(data, model_class, model_parameters, beta, n_init=10):
    best_joint_loss = inf
    best_state = None
    for t in range(n_init):
        L = model_class.randomModelLosses(data, 1, model_parameters)
        n = L.shape[0]
        _, samples = L.shape
        alpha = beta * samples
        w = weightsForLosses(L[0, :], alpha)
        w_old = zeros(samples)
        s = None
        while not allclose(w, w_old):
            w_old = w
            s = model_class(data, w, modelParameters=model_parameters)
            w = weightsForLosses(s.squaredLosses(), alpha)
        L = s.squaredLosses()
        joint_loss = penalizedMultipleWeightedLoss2(L[newaxis, :], w[newaxis, :], beta * n)
        if best_joint_loss > joint_loss:
            best_joint_loss = joint_loss
            best_state = s
    return best_state


def huber_clustering(X, k, eps):
    n, dim = X.shape
    centers = (X.T[:, sample(range(n), k)]).T
    L = array([squaredColNorm((X - center).T) for center in centers])
    meps = sqrt(L.max()) / 3
    #print("meps is %s" % meps)
    steps = max(3, int(log2(meps / eps)))
    #print("steps is %s" % steps)
    epses = logspace(log10(eps), log10(meps), steps)[::-1]
    #print("epses are %s" % epses)
    r = 0
    best_centers = None
    for ceps in epses:
        old_centers = zeros((k, dim))
        #print("Using ceps = %s" % ceps),
        while not allclose(centers, old_centers):
            r += 1
            #print("."),
            if r > 100:
                #print "max abs diff in centers: %s" % np.abs(old_centers - centers).max()
                r = 0
            old_centers = centers
            L = array([squaredColNorm((X - center).T) for center in centers])
            best_centers = L.argmin(axis=0)
            diffs = centers[best_centers, :] - X
            fixed = soft_shrink_rows(diffs, ceps) + X
            centers = array([fixed[best_centers == j, :].mean(axis=0) for j in range(k)])
        #print(":")
    #print("Done")
    return centers, best_centers


def location_data_one_sided_noise(d, noise_offset, noisy_proportion=0.1):
    num_points = 1000
    data_per_column = randn(d, num_points) / sqrt(d)
    first_outlier = int(floor((1 - noisy_proportion) * num_points))
    data_per_column[0, first_outlier:num_points] *= 10
    data_per_column[0, first_outlier:num_points] += noise_offset
    return data_per_column


def soft_shrink_rows(X, length):
    """ Reduce the Euclidean norm of rows of X by length (down to zero). """
    curr_row_lengths = np.sqrt(squaredColNorm(X.T))
    # trivial rows are unmodified:
    tr = logical_not(curr_row_lengths > 0)
    divisors = curr_row_lengths
    divisors[tr] = 1
    unit_length_rows = (X.T / divisors).T
    desired_lengths = nonNegativePart(curr_row_lengths - length)
    return (unit_length_rows.T * desired_lengths).T


def location_estimation_minimal_experiment(d):
    data_point_per_column = location_data_one_sided_noise(d, 1000)
    _, n = data_point_per_column.shape
    alpha = 3 * n

    rw_centers, rw_labels = si.clustering(data_point_per_column.T, 1, alpha, n_init=10)
    rw2_centers = location_estimation2(data_point_per_column)
    hb_centers, hb_labels = huber_clustering(data_point_per_column.T, 1, 10)

    rwc = rw_centers[0][0]
    rw2c = rw2_centers[0]
    hbc = hb_centers[0][0]

    print("90% of data has mean of zero, stddev of 1. 10% of data have mean 1000.")
    print("Values of different estimators:")

    results = [("rw 3", rwc),
               ("rw2 3", rw2c),
               ("hb 10", hbc),
               ("mean", data_point_per_column[0, :].mean()),
               ("median", np.median(data_point_per_column)),
               ("inlier mean", np.mean(data_point_per_column[0, :900])),
               ("inlier median", np.median(data_point_per_column[0, :900]))]

    for n, e in results:
        print("Alg %s estimates: %s" % (n, e))


def location_estimation_experiment_results(d, noise_proportions, noise_offsets):
    def make_task(noise_prop, noise_offset):
        data_point_per_column = location_data_one_sided_noise(d, noise_offset=noise_offset, noisy_proportion=noise_prop)
        gt = zeros(d)
        return data_point_per_column, gt

    def score(x, xgt):
        return norm(x-xgt)

    def huber(data_point_per_column):
        hb_centers, hb_labels = huber_clustering(data_point_per_column.T, 1, 0.01)
        return hb_centers[0]

    def weighted(data_point_per_column):
        rw2_center = location_estimation2(data_point_per_column)
        return rw2_center

    def regular(data_point_per_column):
        return data_point_per_column.mean(axis=1)

    return experiment_matrix_results(make_task, noise_proportions, noise_offsets,
                                     [(huber, "Min. Huber Loss."),
                                     (weighted, "RW loss."),
                                     (regular, "std. averaging.")],
                                     score)


def location_estimation_experiment_results_old(d, noise_offsets, noise_proportions):
    n_props = noise_proportions.shape[0]
    n_dists = noise_offsets.shape[0]
    resrw = ones((n_dists, n_props)) * inf
    reshb = ones((n_dists, n_props)) * inf
    for ip, noise_prop in enumerate(noise_proportions):
        for io, offset in enumerate(noise_offsets):
            data_point_per_column = location_data_one_sided_noise(d, noise_offset=offset, noisy_proportion=noise_prop)
            _, n = data_point_per_column.shape

            rw2_centers = location_estimation2(data_point_per_column)
            hb_centers, hb_labels = huber_clustering(data_point_per_column.T, 1, 0.01)

            rw2c = rw2_centers[0]
            hbc = hb_centers[0][0]
            resrw[io, ip] = rw2c
            reshb[io, ip] = hbc
    return resrw, reshb


def location_estimation_experiment():
    n_props = 10
    noise_offsets = array([0, 10, 100, 1000])
    noise_proportions = linspace(0, 0.45, n_props)
    resrw, reshb = location_estimation_experiment_results(10, noise_offsets, noise_proportions)
    plt.imshow(resrw)
    plt.title("Given mixture of data source (N(0,1/sqrt(d)) and " +
              "noisy source (N(mu,10/sqrt(d))), estimate data source mean.")
    plt.xlabel("Proportion of noisy source.")
    plt.ylabel("Distance between sources.")
