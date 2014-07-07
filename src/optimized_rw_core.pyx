import numpy as np
cimport numpy as np

ctypedef np.double_t DTYPE_t
ctypedef np.intp_t ITYPE_t


def lambdaAndWinners1(L, ts):
    arr = (ts - L.T).T

    # Find the best model to explain each point, and base weights there
    idxes = arr.argmax(0)
    maxes = arr.max(0)
    return maxes, idxes

def lambdaAndWinners2(L, ts):
    cdef int n
    cdef int k
    k,n = L.shape
    idxes = np.zeros(n,dtype='int')
    maxes = np.zeros(n)
    cdef int i
    for i in range(n):
        adjusted = ts - L[:,i]
        idxes[i] = adjusted.argmax()
        maxes[i] = adjusted.max()

    return maxes, idxes

cimport cython
@cython.boundscheck(False)
def lambdaAndWinners(L, np.ndarray[DTYPE_t, ndim=1] ts):
    cdef np.ndarray[DTYPE_t, ndim=2] Lb = L
    cdef int n
    cdef int k
    k,n = L.shape
    cdef np.ndarray[ITYPE_t, ndim=1] idxes = np.zeros(n,dtype=np.intp)
    cdef np.ndarray[DTYPE_t, ndim=1] maxes = np.zeros(n)
    cdef unsigned int i
    cdef unsigned int j
    cdef int colWin
    cdef double colMax
    cdef double curr
    cdef double low = -np.inf
    for i in range(n):
        colMax = low
        colWin = -1
        for j in range(k):
            curr = ts[j] - Lb[j,i]
            if curr > colMax:
               colMax = curr
               colWin = j
        idxes[i] = colWin
        maxes[i] = colMax

    return maxes, idxes

@cython.boundscheck(False)
def update_win_lam(np.ndarray[DTYPE_t, ndim=1] lam_tu,
                   np.ndarray[ITYPE_t, ndim=1] win_tu,
                   np.ndarray[DTYPE_t, ndim=1] lam_r,
                   np.ndarray[ITYPE_t, ndim=1] win_r):
    cdef unsigned int i
    cdef double lr
    cdef int wr
    n = lam_tu.shape[0]
    if ((lam_r.shape[0] is not n) or
        (win_tu.shape[0] is not n) or
        (win_r.shape[0] is not n)):
        print("Inconsistent lengths in update_win_lam.")
    for i in range(n):
        lr = lam_r[i]
        if lam_tu[i] < lr:
            wr = win_r[i]
            win_tu[i] = wr
            lam_tu[i] = lr


