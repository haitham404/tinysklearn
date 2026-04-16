import numpy as np

class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.class_to_index = {}

    def fit(self, y):
        y = np.array(y)
        self.classes_ = np.unique(y)
        for i in range(len(self.classes_)):
            self.class_to_index[self.classes_[i]] = i
        return self

    def transform(self, y):
        y = np.array(y)
        result = []
        for i in range(len(y)):
            if y[i] not in self.class_to_index:
                raise ValueError("Found unseen label")
            result.append(self.class_to_index[y[i]])
        return np.array(result)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    def inverse_transform(self, y):
        y = np.array(y)
        result = []
        for i in range(len(y)):
            result.append(self.classes_[y[i]])
        return np.array(result)
