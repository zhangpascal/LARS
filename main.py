import numpy as np
from gen_data import gen_data

n = 10
p = 20
seed = 42

X, y, beta = gen_data(n, p, seed)

print(X)
print(y)
