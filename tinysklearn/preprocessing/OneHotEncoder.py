import numpy as np

class OneHotEncoder:
    def __init__(self, drop_first=False):
        self.categories_ = None
        self.category_to_index = {}
        self.drop_first = drop_first

    def fit(self, y):
        y = np.array(y)

        self.categories_ = np.unique(y)

        for i in range(len(self.categories_)):
            self.category_to_index[self.categories_[i]] = i

        return self

    def transform(self, y):
        y = np.array(y)

        n_samples = len(y)
        n_categories = len(self.categories_)

        # لو drop_first نقلل column
        if self.drop_first:
            result = np.zeros((n_samples, n_categories - 1))
        else:
            result = np.zeros((n_samples, n_categories))

        for i in range(n_samples):
            if y[i] not in self.category_to_index:
                raise ValueError("Found unseen category")

            j = self.category_to_index[y[i]]

            if self.drop_first:
                if j > 0:          # skip first category
                    result[i][j - 1] = 1
            else:
                result[i][j] = 1

        return result

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
