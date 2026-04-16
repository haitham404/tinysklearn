import numpy as np
from tinysklearn.optimizers import gradient_descent

class LogisticRegression:
    def __init__(self,
         fit_intercept=True,
         lr=0.01,
         n_iters=1000,
         tol=1e-6,
         verbose=False ):

        self.fit_intercept = fit_intercept
        self.lr = lr
        self.n_iters = n_iters
        self.tol = tol
        self.verbose = verbose

        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    def _validate_input(self, X, y=None):
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y is not None:
            y = np.array(y).reshape(-1, 1)

            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have same number of samples.")

        return X, y

    def _add_intercept(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def _sigmoid(self, i):
        return 1 / (1 + np.exp(-i))

    def fit(self, X, y):
        X, y = self._validate_input(X, y)

        if self.fit_intercept:
            X = self._add_intercept(X)

        self._fit_gd(X, y)

        return self

    def _fit_gd(self, X, y):
        n_features = X.shape[1]
        initial_params = np.zeros(n_features, dtype=float)
        y_vec = y.flatten()

        def fun_driv(params):
            z = np.dot(X, params)
            y_pred = self._sigmoid(z)

            grad = (1 / X.shape[0]) * np.dot(X.T, (y_pred - y_vec))
            return grad

        params = gradient_descent(
            fun_driv,
            initial_params,
            step_size=self.lr,
            precision=self.tol,
            max_iter=self.n_iters
        )

        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:].flatten()
        else:
            self.intercept_ = 0.0
            self.coef_ = params.flatten()

    def predict_proba(self, X):
        X, _ = self._validate_input(X)

        if self.fit_intercept:
            z = np.dot(X, self.coef_) + self.intercept_
        else:
            z = np.dot(X, self.coef_)

        return self._sigmoid(z)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)