import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linspace, arange, log10, logspace, ones, inf, ceil, sqrt, floor, zeros, diag, eye, dot
from numpy.random import randn
from numpy.linalg import norm, svd
from scipy import stats
from sklearn import datasets, linear_model

mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True


#Data generation subroutine
#-------------------------------------------------------------------------
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

#--------------------------------------------------------------------------

#Assessing result
#--------------------------------------------------------------------------
def angleCosinesBetweenBases(baseForFirst, baseForSecond):
    prod = dot(baseForFirst.transpose(), baseForSecond)
    _, s, _ = svd(prod, full_matrices=False)
    return s
#--------------------------------------------------------------------------

#Computing result main function
#--------------------------------------------------------------------------
def pca_experiment_matrix_results(big_dim, d, samples, noisy_proportions, disturbance_levels):
    def make_task(noise_prop, dist_lvl):
        X, gt_basis = annoying_pca_data(d, big_dim, samples, noise_prop, 0.1, dist_lvl)
        return X, gt_basis

    def score(basis, gt_basis):
        return norm(1 - angleCosinesBetweenBases(basis[:, :big_dim], gt_basis[:, :big_dim]))

    def cstep(X,basis,h):
        #compute loss for each data point
        losses=zeros(samples)
        for i,loss in enumerate(losses):
            loss=norm(X[i,:])**2-norm(basis[:,:big_dim].T.dot(X[i,:]))**2

        #find the h data points with least loss
        new_set=np.argpartition(losses,h)[:h]

        #find new basis
        new_X=X[new_set,:]
        _,_,new_basis=svd(new_X,full_matrices=True)

        return (new_basis, new_set)

    def LTS_PCA(X):
        #initial subset selection
        n,p=X.shape
        h=(n+p+1)/2
        old_set=np.random.choice(range(n),h,replace=False)
        _,_,old_basis=svd(X[old_set,:],full_matrices=True)

        #begin iteration of cstep until convergence
        iter_times=1
        while True:
            new_basis, new_set=cstep(X,old_basis,h)
            iter_times+=1
            if all(np.sort(new_set)==np.sort(old_set)):
                break
            old_set=new_set
            old_basis=new_basis

        print 'Number of iteration: %d'%iter_times

        return new_basis

    return experiment_matrix_results(make_task, noisy_proportions, disturbance_levels,
                                     [(LTS_PCA, "Least Trimmed Squares")],score)
#--------------------------------------------------------------------------

#Results formulation subroutine
#--------------------------------------------------------------------------
def experiment_matrix_results(make_data, row_param_values, col_param_values, methods, summarize):
    n_cols = col_param_values.shape[0]
    n_rows = row_param_values.shape[0]
    result_matrices = [ones((n_rows, n_cols)) * inf for _ in methods]
    for ip, offset in enumerate(col_param_values):
        for io, noise_prop in enumerate(row_param_values):
            data, gt = make_data(noise_prop, offset)
            for i, (m, _) in enumerate(methods):
                # print 'offset strength:'+repr(offset)+' noise proportion'+repr(noise_prop)
                estimate = m(data)
                result_matrices[i][io, ip] = summarize(estimate, gt)
    method_names = [n for (m, n) in methods]
    return result_matrices, method_names
#--------------------------------------------------------------------------

#Result displaying subroutine
#--------------------------------------------------------------------------
def display_result_matrices(matrices, method_names, row_vals, col_vals, ylabel, xlabel):
    matrices = [np.log10(m) for m in matrices]
    high = max([m.max() for m in matrices])
    low = min([m.min() for m in matrices])
    n_methods = len(method_names)

    fig, axes = plt.subplots(nrows=1, ncols=n_methods)

    for i, (m, n) in enumerate(zip(matrices, method_names)):
        plt.subplot(1, n_methods, i+1)
        im = plt.imshow(m, cmap=plt.cm.gray_r, vmin=low, vmax=high, interpolation='nearest')
        plt.title(n)
        plt.xticks(arange(col_vals.shape[0]), ["%.2f" % cv for cv in col_vals], rotation=70)
        plt.xlabel(xlabel)  # Can share via:
        # http://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
        if i == 0:
            plt.yticks(arange(row_vals.shape[0]), ["%.2f" % rv for rv in row_vals])
            plt.ylabel(ylabel)
        else:
            plt.yticks([])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.4, 0.03, 0.2])
    fig.colorbar(im, cax=cbar_ax)
#---------------------------------------------------------------------------

#Start experiment, data generation as before
#--------------------------------------------------------------------------
d = 20
big_dim = 4
samples = 1000

noisy_proportions = linspace(0, 0.45, 5) #linspace(0, 0.45, 10)
disturbance_levels = logspace(-1, 1, 5) #logspace(-2, 1, 9)
n_noisy_props=noisy_proportions.shape[0]
n_alt_src_strs=disturbance_levels.shape[0]

#conduct several experiment and take average
resmat=[np.zeros((n_noisy_props,n_alt_src_strs))]
method_names=[None]
for i in range(10):
    mat_inc, method_names = pca_experiment_matrix_results(big_dim, d, samples, noisy_proportions, disturbance_levels)
    for j,_ in enumerate(resmat):
        resmat[j]+=mat_inc[j]

for mat in resmat:
    mat/=10

joblib.dump((resmat, method_names, noisy_proportions, disturbance_levels), "pca_results.jl", compress=3)

(resmat, method_names, noisy_proportions, disturbance_levels) = joblib.load("pca_results.jl")
display_result_matrices(resmat, method_names,
                            noisy_proportions, disturbance_levels,
                            "Proportion of noisy points.", "Strength of alt. source.")

plt.savefig("pcaResults.pdf", bbox_inches='tight')
plt.show()
