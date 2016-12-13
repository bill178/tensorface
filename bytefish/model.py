import numpy as np
import numpy.linalg as LA
from bytefish.subspace import *

class EigenFace(object):
    def __init__(self, arg_X, arg_y):
        self.projections = []
        self.W = []
        self.mu = []
        if (arg_X is not None) and (arg_y is not None):
            self.compute(arg_X, arg_y)

    def compute(self, arg_X, arg_y):
        self.W, self.mu, cache = pca(arg_X)
        # store labels
        self.y = arg_y
        # store projections
        for xi in arg_X:
            self.projections.append(project(self.W, xi.reshape(1, -1), self.mu))

    def predict(self, arg_X):
        minDist = np.finfo('float').max
        minClass = -1
        Q = project(self.W, arg_X.reshape(1,-1), self.mu)
        for i in xrange(len(self.projections)-1):
            dist = LA.norm(self.projections[i] - Q)
            if dist < minDist:
                minDist = dist
                minClass = self.y[i]
        return minClass