import numpy as np

def pre_proc(X):
    X = np.array(X) 
    
    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    X = (X-m)/std  
    return X

def gen_data(n, p, num_b, seed):
    
    mean_mat = np.zeros(p) 
    cov_mat = np.eye(p)

    rng = np.random.default_rng(seed) #reproducibility
    X = rng.multivariate_normal(mean_mat, cov_mat, n) #data 
    
    X = pre_proc(X) #standardisation 
    
    beta = np.zeros(p) 
    
    beta[rng.choice(p, num_b, replace = False)] = 1  #selection of random beta 
    
    y = X@beta
    
    return X, y, beta