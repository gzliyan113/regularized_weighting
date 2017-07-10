import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, linspace, arange, log10, logspace, ones, inf, ceil
from numpy.random import randn
from numpy.linalg import norm
from scipy import stats
from sklearn import datasets, linear_model

#Data generation subroutine
#-------------------------------------------------------------------------
def create_regression_task(d, n, rns,  noise_strength, alt_source_strength, noisy_proportion=0.1):
    # print alt_source_strength, noisy_proportion
    # print norm(rns)
    X = randn(d, n) / np.sqrt(d)
    rgt = np.zeros(d)
    rgt[0] = 1
    #print("rns = %s" % rns)
    y = rgt.dot(X) + randn(n) / 10
    first_outlier = int(ceil((1 - noisy_proportion) * n))
    #print("first_outlier = %s" % first_outlier)

    outlier_mods = rns.dot(X[:, first_outlier:]) * alt_source_strength + \
    randn(n - first_outlier) * noise_strength
#print("outlier_mods %s" % outlier_mods)
    y[first_outlier:] += outlier_mods
    return X, y, rgt

#--------------------------------------------------------------------------

#Computing result main function
#--------------------------------------------------------------------------
def linear_regression_experiment_matrix_results2(d, n, rns, noisy_props, alt_src_strengths):
    def make_task(noisy_prop, alt_src_str):
        X, y, rgt = create_regression_task(d, n, rns,  0.1, alt_src_str, noisy_proportion=noisy_prop)
        return (X, y), rgt

    def score(r, rgt):
        return norm(r - rgt)

    def cstep(data,old_set,old_model,h):
        '''data: 2-dimenional n*(p+1) array;
           old_set: the index of previous subset;
           old_cpef: previous regression model coefficients
        '''
        #first compute residual from previous model
        n=data.shape[0]
        covariates=data[:,:-1]
        obs=data[:,-1]
        prediction=old_model.predict(covariates).reshape(n)
        residual=abs(obs-prediction)

        #find the h data points with least residual
        new_set=np.argpartition(residual,h)[:h]

        #fit the new model to the new subset
        new_covariates=data[new_set,:-1]
        new_obs=data[new_set,-1]
        regr = linear_model.LinearRegression()
        regr.fit(new_covariates, new_obs)

        return((new_set,regr))

    def LST_LS(data):
        '''
            computing least trimmed square estimator
            data: n*p dimension array
        '''
        X,y=data
        X=np.transpose(X)
        y=y[:,np.newaxis]
        data=np.concatenate((X,y),axis=1)

        n,p=data.shape
        h=(n+p+1)/2
        X=data[:,:-1]
        Y=data[:,-1]
        Y=Y[:,np.newaxis]
        model=linear_model.LinearRegression(fit_intercept=False)

        #initial subset selection
        subset=np.random.choice(range(n),h,replace=False)
        model.fit(X[subset,:],Y[subset,:])

        #begin iteration of c-step, until process converge
        iter_times=1
        while True:
            newset, new_model=cstep(data,subset,model,h)
            iter_times+=1
            if all(np.sort(newset)==np.sort(subset)):
                break
            subset=newset
            model=new_model

        # plt.plot(X,Y,'ro')
        # plt.plot(X,model.predict(X),ls='-',color='blue')
        # plt.show()
        print 'Number of iteration: %d'%iter_times

        return new_model.coef_

    return experiment_matrix_results(make_task, noisy_props, alt_src_strengths,
                                     [(LST_LS, "Least Trimmed Square")],score)
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
alt_src_strengths = logspace(-1, 1, 9)
noisy_props = linspace(0, 0.45, 10)
n_alt_src_strs = alt_src_strengths.shape[0]
n_noisy_props = noisy_props.shape[0]

#generate random permutation signal
d=10
n=1000
rns = randn(d) / np.sqrt(d)

#conduct several experiment and take average
resmat=[np.zeros((n_noisy_props,n_alt_src_strs))]
method_names=[None]
for i in range(10):
    mat_inc, method_names = linear_regression_experiment_matrix_results2(d, n, rns, noisy_props, alt_src_strengths)
    for j,_ in enumerate(resmat):
        resmat[j]+=mat_inc[j]

for mat in resmat:
    mat/=10
joblib.dump((resmat, method_names, noisy_props, alt_src_strengths), "linear_results.jl", compress=3)

#Plot experiment result
(resmat, method_names, noisy_props, alt_src_strengths) = joblib.load("linear_results.jl")
display_result_matrices(resmat, method_names,
                            noisy_props, alt_src_strengths,
                            "Proportion of noisy points.", "Strength of alternative source.")

plt.savefig("linearRegressionResults.pdf", bbox_inches='tight')
plt.show()
