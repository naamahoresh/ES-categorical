from scipy import *
from scipy.linalg import norm, pinv
from scipy.spatial.distance import hamming
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from decimal import *
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        #self.beta = 1/((2*(indim**2)))
        self.beta =0.1
        #self.beta = np.ones(numCenters)
        #self.sigma = np.ones(numCenters)
        self.W = random.random((self.numCenters, self.outdim))
        self.W0 = 0


    def _basisfunc(self, c,x):
        assert (len(x) == self.indim), len(x)
        #ans= exp(-self.beta * norm(c-x)**2)
        ans= exp(-self.beta * self.hamming_distance(c,x) ** 2)
        return ans

    def hamming_distance(self, s1, s2):
        assert len(s1) == len(s2)
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] =  self._basisfunc(c, x)
        return G

    def train(self, X, Y, isKmeans=False):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """
        #self.select_beta(X) #update beta

        if (isKmeans): ##choose the canters using Kmean algorithm
            kmeanz = KMeans(n_clusters=self.numCenters).fit(X)
            closest, _ = pairwise_distances_argmin_min(kmeanz.cluster_centers_, X)
            closest = np.array(closest)
            if not (0 in closest): #make sure that the best vector is saved
                closest[0]=0
            self.centers=[X[i, :] for i in closest]

        else: ##choose the canters randomly
            rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
            if not (0 in rnd_idx): #make sure that the best vector is saved
                rnd_idx[0]=0 #naama
            self.centers = [X[i, :] for i in rnd_idx]

        G = self._calcAct(X)        # calculate activations of RBFs

        self.W0 = mean(Y)
        self.W = dot(pinv(G),Y-self.W0)        # calculate output weights (pseudoinverse)
        #self.W = dot(pinv(G), Y)       # calculate output weights (pseudoinverse)


    def test(self, X):
        #self.select_beta(X)

        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = dot(G, self.W)+ self.W0
        #Y = dot(G, self.W)

        return Y

    def select_beta(self,X): # choose beta by http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
        for ci, c in enumerate(self.centers):
            self.sigma[ci] = (sum(self.hamming_distance(c, point) for point in X)) * (1 / len(X))
            self.beta[ci] = 1 / Decimal(2 * (self.sigma[ci] ** 2))