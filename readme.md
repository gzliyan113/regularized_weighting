This is research code, no guarantees are implied.

In particular, if the install instructions do not work for you, please let us know.

Issues
============

- Does not support class imbalance (eta_j)
- Serious dependency on cvxopt, which is hard to install
- Corresponds to alpha formulation to alpha, not beta (no preweighting of data)


Install instructions
============

# (generic, see for particular platforms below):

- Download and install: http://cvxopt.org
- Download and install: http://scikit-learn.org/stable/install.html 
- Download and install: https://nose.readthedocs.org/en/latest/
- git clone https://github.com/daniel-vainsencher/regularized_weighting.git
- cd src
- python setup.py build_ext --inplace
- nosetests

# Ubuntu (and likely other Debian-derived, thanks to Nadav Maoz):

sudo apt-get install atlas-base-dev python-dev glpk build-essential python-dev python-numpy python-setuptools python-scipy libatlas-dev libatlas3-base python-matplotlib python-pip git cython

sudo pip install -U scikit-learn
sudo pip install nose

# download cvxopt from https://github.com/cvxopt/cvxopt/archive/1.1.6.tar.gz and unpack
# in the cvxopt dir:
sudo python setup.py install

sudo git clone https://github.com/daniel-vainsencher/regularized_weighting.git #(in appropriate directory)
# enter src directory therein
sudo python setup.py build_ext --inplace
nosetests


# MacOS (thanks to François Schnitzler):

sudo port install python27 glpk py27-numpy py27-setuptools py27-scipy py27-scikit-learn py27-matplotlib py27-pip py27-cython atlas py27-nose py27-cvxopt

#install cvxopt (can also be done using macport)
# This was difficult. Problem was linking to the correct atlas / lapack libraries. I ended up changing the name of the libraries in setup.py to "tatlas"

#change setup.py to configure numpy headers inclusion
