import numpy as np
from numpy import linalg as LA

def pca(arg_X):
    n,d = arg_X.shape
    
    # math expectation
    mu = arg_X.mean(axis=0)
    arg_X = arg_X - mu
    if n > d:
        C = np.dot(arg_X.T, arg_X)
        val_eigen, vec_eigen = LA.eigh(C)
    else:
        C = np.dot(arg_X, arg_X.T)
        val_eigen, vec_eigen = LA.eigh(C)
        vec_eigen = np.dot(arg_X.T, vec_eigen)
        for i in xrange(n):
            vec_eigen[:,i] = vec_eigen[:,i] / LA.norm(vec_eigen[:,i])
    # or simply perform an economy size decomposition
    # vec_eigen, val_eigen, variance = LA.svd(arg_X.T, full_matrices=False)
    # sort vec_eigen descending by their eigenvalue
    idx = np.argsort(-val_eigen)
    val_eigen = val_eigen[idx]
    vec_eigen = vec_eigen[:,idx]

    # vec_eigen.shape = (d, n)
    # val_eigen.shape = (n, )

    # evaluate the number of principal components
    # needed to represent 95% Total variance
    eigen_sum = np.sum(val_eigen)
    csum, k95 = 0, 0
    for i in xrange(len(val_eigen)):
        csum += val_eigen[i]
        tv = csum / eigen_sum
        if tv > 0.95:
            k95 = i
            break
    if (k95 <= 0) or (k95 > n):
        k95 = n

    # select only k95
    val_eigen = val_eigen[0:k95].copy()
    W = vec_eigen[:,0:k95].copy()

    cache = [val_eigen, k95]
    return W, mu, cache

def project(arg_W, arg_X, mu=None):
    if mu is None:
        return np.dot(arg_X, arg_W)
    return np.dot(arg_X - mu, arg_W)

def reconstruct(arg_W, arg_Y, mu=None):
    if mu is None:
        return np.dot(arg_Y, arg_W.T)
    return np.dot(arg_Y, arg_W.T) + mu
