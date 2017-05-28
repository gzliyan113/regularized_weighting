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
from numpy import array, linspace, arange, log10

# <codecell>

n_props = 10
noise_proportions = linspace(0, 0.45, n_props)
noise_offsets = array([0, 0.1, 1, 10, 100, 1000])

# <codecell>

resmat, method_names, method_times = sms.location_estimation_experiment_results(10, noise_proportions, noise_offsets)

joblib.dump((resmat, method_names, method_times, noise_proportions, noise_offsets), "loc_est_results.jl", compress=3)


# <codecell>

(resmat, method_names, method_times, noise_proportions, noise_offsets) = joblib.load("loc_est_results.jl")
sms.display_result_matrices(resmat, method_names,
                            noise_proportions, noise_offsets,
                            "Proportion of noisy points.", r"$\left\Vert a - b \right\Vert_2$")
                            #"Distance of alternative source."

plt.savefig("locationEstimationResults.pdf", bbox_inches='tight')
plt.show()

# <codecell>
