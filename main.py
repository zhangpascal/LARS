import numpy as np
from gen_data import gen_data

n = 3
p = 5
seed = 42

X = gen_data(n, p, seed)

print(np.cov(X, rowvar=False))