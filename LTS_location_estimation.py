import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linspace, arange, log10, logspace, ones, inf, ceil, sqrt, floor, zeros
from numpy.random import randn
from numpy.linalg import norm
from scipy import stats
from sklearn import datasets, linear_model

mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True


#Data generation subroutine
#-------------------------------------------------------------------------
def location_data_one_sided_noise(d, noise_offset, noisy_proportion=0.1):
    num_points = 1000
    data_per_column = randn(d, num_points) / sqrt(d)
    first_outlier = int(floor((1 - noisy_proportion) * num_points))
    data_per_column[0, first_outlier:num_points] *= 10
    data_per_column[0, first_outlier:num_points] += noise_offset
    return data_per_column

#--------------------------------------------------------------------------

#Computing result main function
#--------------------------------------------------------------------------
def location_estimation_experiment_results(d, noise_proportions, noise_offsets):
    def make_task(noise_prop, noise_offset):
        data_point_per_column = location_data_one_sided_noise(d, noise_offset=noise_offset, noisy_proportion=noise_prop)
        gt = zeros(d)
        return data_point_per_column, gt

    def score(x, xgt):
        return norm(x-xgt)

    def cstep(data_point_per_column,old_set,old_center,h):
        n=data_point_per_column.shape[1]
        dist_mat=data_point_per_column-np.tile(old_center[:,np.newaxis],(1,n))
        dist_vec=np.array([norm(dist_mat[:,i]) for i in range(n)])

        #find the h data points with least distance
        new_set=np.argpartition(dist_vec,h)[:h]

        return (np.mean(data_point_per_column[:,new_set],axis=1), new_set)


    def LTS_LOC(data_point_per_column):
        #initial subset selection
        p,n=data_point_per_column.shape
        h=(n+2)/2
        old_set=np.random.choice(range(n),h,replace=False)
        old_center=np.mean(data_point_per_column[:,old_set],axis=1)

        #begin iteration of c-step, until process converge
        iter_times=1
        while True:
            new_center, new_set=cstep(data_point_per_column,old_set,old_center,h)
            iter_times+=1
            if all(np.sort(new_set)==np.sort(old_set)):
                break
            old_set=new_set
            old_center=new_center

        # plt.plot(data_point_per_column[0,:],data_point_per_column[1,:],'ro')
        # plt.plot(new_center[0],new_center[1],'go',markersize=12)
        # plt.show()
        print 'Number of iteration: %d'%iter_times

        return new_center


    return experiment_matrix_results(make_task, noise_proportions, noise_offsets,
                                     [(LTS_LOC, "Least Trimmed Squares")],score)
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
n_props = 10
noise_proportions = linspace(0, 0.45, n_props)
noise_offsets = array([0, 10, 100, 1000])
n_noisy_props=noise_proportions.shape[0]
n_alt_src_strs=noise_offsets.shape[0]

#conduct several experiment and take average
resmat=[np.zeros((n_noisy_props,n_alt_src_strs))]
method_names=[None]
for i in range(10):
    mat_inc, method_names = location_estimation_experiment_results(10, noise_proportions, noise_offsets)
    for j,_ in enumerate(resmat):
        resmat[j]+=mat_inc[j]

for mat in resmat:
    mat/=10

joblib.dump((resmat, method_names, noise_proportions, noise_offsets), "loc_est_results.jl", compress=3)

#Plot experiment result
(resmat, method_names, noise_proportions, noise_offsets) = joblib.load("loc_est_results.jl")
display_result_matrices(resmat, method_names,
                            noise_proportions, noise_offsets,
                            "Proportion of noisy points.", r"$\left\Vert a - b \right\Vert_2$")

plt.savefig("locationEstimationResults.pdf", bbox_inches='tight')
plt.show()
