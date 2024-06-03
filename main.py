import numpy as np
from gen_data import gen_data
from sklearn import linear_model as lin_mod

n = 300
p = 1000
num_b = 10
seed = 42

X, y, beta = gen_data(n, p, num_b, seed)

reg = lin_mod.Lars()
reg.fit(X, y)

a = reg.coef_>0
b = beta>0

#print(a, b)

print(np.sum((a == b)))


