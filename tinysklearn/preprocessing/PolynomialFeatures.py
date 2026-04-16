import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def fit_transform(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        result = [np.ones(n_samples)] if self.include_bias else []

        # add original features (degree 1)
        for feature in range(n_features):
            result.append(X[:, feature])

        # add higher degree features
        if self.degree >= 2:
            for deg in range(2, self.degree + 1):
                comb = combinations_with_replacement(range(n_features), deg)
                for indices in comb:
                    temp = np.ones(n_samples)
                    for idx in indices:
                        temp *= X[:, idx]
                    result.append(temp)

        return np.column_stack(result)
