This is research code; no guarantees are implied, and the best information about it is currently in the paper:

Vainsencher, Daniel, Shie Mannor, and Huan Xu. "Learning Multiple Models via Regularized Weighting."
Advances in Neural Information Processing Systems. 2013.

We would love to know about it if you use our method, and please cite the above paper in any publications.

In particular, if the install instructions do not work for you, please let us know.

Issues
============

- Corresponds to alpha formulation to alpha, not beta (no preweighting of data)

Build and test status
============

Not all tests pass on Travis yet; the button below will tell you what, and feel free to contact us for help.
[![Build Status](https://travis-ci.org/daniel-vainsencher/regularized_weighting.svg?branch=master)](https://travis-ci.org/daniel-vainsencher/regularized_weighting)

Install instructions
============

Follow the scripts in .travis.yaml, which are automatically tested. To summarize:

- Install the miniconda script
- Use miniconda to install all the dependencies
- Build the cython code
- Run the tests