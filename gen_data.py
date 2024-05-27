import numpy as np

def gen_data(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p))
    return X