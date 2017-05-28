# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
import joblib

import singleModelSpike as sms
from numpy import array, linspace, arange, log10, logspace

d = 20
big_dim = 4
samples = 1000

noisy_proportions = linspace(0, 0.45, 5) #linspace(0, 0.45, 10)
disturbance_levels = logspace(-1, 1, 5) #logspace(-2, 1, 9)

resmat, method_names, time_mat = sms.pca_experiment_matrix_results(big_dim, d, samples, noisy_proportions, disturbance_levels)
joblib.dump((resmat, method_names, time_mat, noisy_proportions, disturbance_levels), "pca_results.jl", compress=3)
# <codecell>
import joblib

import matplotlib as mpl
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import singleModelSpike as sms

(resmat, method_names, time_mat, noisy_proportions, disturbance_levels) = joblib.load("pca_results.jl")
sms.display_result_matrices(resmat, method_names,
                            noisy_proportions, disturbance_levels,
                            "Proportion of noisy points.", "Strength of alt. source.")

plt.savefig("pcaResults.pdf", bbox_inches='tight')
plt.show()

sms.display_result_matrices(time_mat, method_names,
                            noisy_proportions, disturbance_levels,
                            "Proportion of noisy points.", "Str. of alt. source.")

plt.savefig("pcaTimes.pdf", bbox_inches='tight')
plt.show()
