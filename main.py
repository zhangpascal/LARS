import numpy as np
import matplotlib.pyplot as plt
from gen_data import gen_data
from sklearn import linear_model as lin_mod


n = 500
p = 1000
num_b = 40
seed = 42

X, y, beta = gen_data(n, p, num_b, seed)

reg = lin_mod.Lars()
reg.fit(X, y)

a = reg.coef_>0 #selection of actives 
b = beta>0

fig = plt.figure()
#plt.subplot(2,1,1)
plt.plot(a,"b+")
#plt.subplot(2,1,2)
plt.plot(b,"rx")
plt.show()
