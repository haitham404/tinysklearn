import numpy as np

def train_val_test_split(X, y, val_size=0.1, test_size=0.2, random_state=None, shuffle=True):
    X = np.array(X)
    y = np.array(y)

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    test_size = int(n_samples * test_size)
    val_size = int(n_samples * val_size)
    train_size = n_samples - val_size - test_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test