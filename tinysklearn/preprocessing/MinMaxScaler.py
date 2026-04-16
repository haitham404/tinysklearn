import numpy as np

class MinMaxScaler:

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        X = np.array(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
