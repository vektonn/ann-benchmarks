from __future__ import absolute_import

import numpy
import sklearn.preprocessing

import faiss
from ann_benchmarks.algorithms.faiss import Faiss


class FaissBruteForce(Faiss):
    def __init__(self, metric):
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError("FaissBruteForce doesn't support metric %s" % metric)
        self._metric = metric
        self.name = 'FaissBruteForce()'

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != numpy.float32:
            X = X.astype(numpy.float32)

        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X)
