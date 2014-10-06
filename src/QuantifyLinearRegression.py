# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#%matplotlib inline
import joblib
import matplotlib as mpl
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True


import singleModelSpike as sms
import matplotlib.pyplot as plt
from numpy import array, linspace, arange, log10, logspace

# <codecell>
#noise_strengths = logspace(-3, 3, 11)
alt_src_strengths = logspace(-1, 1, 9) #array([0,0,0,0,0,0,0,0])
noisy_props = linspace(0, 0.45, 10)
#n_noise_strs = noise_strengths.shape[0]
n_alt_src_strs = alt_src_strengths.shape[0]
n_noisy_props = noisy_props.shape[0]
resmat, method_names = sms.linear_regression_experiment_matrix_results2(10, 1000, noisy_props, alt_src_strengths)
#resmat = sms.linear_regression_experiment_matrix_results(10, 1000, noise_strengths, alt_src_strengths)
joblib.dump((resmat, method_names, noisy_props, alt_src_strengths), "linear_results.jl", compress=3)


# <codecell>

(resmat, method_names, noisy_props, alt_src_strengths) = joblib.load("linear_results.jl")
sms.display_result_matrices(resmat, method_names,
                            noisy_props, alt_src_strengths,
                            "Proportion of noisy points.", "Strength of alternative source.")

plt.savefig("linearRegressionResults.pdf", bbox_inches='tight')
plt.show()
