import numpy as np

def gen_data(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, p))
    
    beta = np.zeros(p)  #beta vector
    
    beta[np.arange(0,3,1)] = 1  #selection of beta 
    
    y = X@beta
    
    return X, y, beta