import numpy as np

a=np.array([1,2,3])
print a.shape


print np.tile(a[:,np.newaxis],(1,3))
print np.linalg.norm(a)**2
