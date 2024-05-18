from sklearn.datasets import make_regression
from sklearn.datasets import load_diabetes
import lars 
import numpy as np

reg = lars.LARS()

X, y = make_regression(n_samples = 1000000, n_features = 100, n_informative= 100, noise=0.6)
X, y = lars.pre_proc(X, y)

reg.fit(X, y)
R2 = 1-np.sum((y-np.dot(X,reg.coef))**2)/np.sum((y-np.mean(y))**2)
print(reg.time)
print(R2)

