import numpy as np

def gradient_descent(fun_driv, initial_state, step_size=0.001, precision=1e-5, max_iter=1000):
    current = np.array(initial_state, dtype=float)
    itr = 0

    while itr < max_iter:
        gradient = fun_driv(current)
        new = current - step_size * gradient

        if np.linalg.norm(new - current) < precision:
            break

        current = new
        itr += 1

    return current