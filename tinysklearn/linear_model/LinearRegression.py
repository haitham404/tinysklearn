import numpy as np
from tinysklearn.optimizers import gradient_descent

class LinearRegression:
    def __init__(self,
         fit_intercept=True,
         solver="normal",
         lr=0.01,
         n_iters=1000,
         tol=1e-6,
         verbose=False ):


        self.fit_intercept = fit_intercept
        self.solver = solver
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



    def fit(self, X, y):
        X, y = self._validate_input(X, y)

        if self.fit_intercept:
            X = self._add_intercept(X)

        if self.solver == "normal":
            self._fit_normal(X, y)

        elif self.solver == "gd":
            self._fit_gd(X, y)

        else:
            raise ValueError("solver must be 'normal' or 'gd'")
        return self


    def _fit_gd(self, X, y):
        n_features = X.shape[1]
        initial_params = np.zeros(n_features, dtype=float)
        y_vec = y.flatten()

        def fun_driv(params):
            y_pred = np.dot(X, params)
            grad = (2 / X.shape[0]) * np.dot(X.T, (y_pred - y_vec))
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

        # self.loss = compute_loss(params)
        # self.loss_history_.append(self.loss)


    def _fit_normal(self, X, y):
        """
        Fit Linear Regression using the Normal Equation:
            theta = (X^T X)^(-1) X^T y
        """
        XT = X.T
        try:
            theta = np.linalg.inv(np.dot(XT, X)).dot(np.dot(XT, y))
        except np.linalg.LinAlgError:
            # In case X^T X is singular, use pseudo-inverse
            theta = np.dot(np.linalg.pinv(np.dot(XT, X)), np.dot(XT, y))

        # Convert to 1D array if needed
        theta = theta.flatten()

        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        y_pred = np.dot(X, theta)



    def predict(self, X):
        X, _ = self._validate_input(X)

        if self.fit_intercept:
            y_pred = np.dot(X, self.coef_) + self.intercept_
        else:
            y_pred = np.dot(X, self.coef_)

        return y_pred

