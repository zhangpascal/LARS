import numpy as np
import matplotlib.pyplot as plt
from gen_data import gen_data
from sklearn import linear_model as lin_mod
from sklearn.metrics import confusion_matrix


n = 500
p = 1000
num_b = 50
seed = 42
lst = []
Corr= [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1]
Perc= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for corr in Corr:
    for perc in Perc:
        X, y, beta = gen_data(n, p, num_b, corr, perc, seed)

        reg = lin_mod.Lars()
        reg.fit(X, y)

        a = (reg.coef_>0.9)*(reg.coef_<1.1) #selection of actives 
        b = beta > 0.9
        
        tp, fn, fp, tn = confusion_matrix(b,a, labels=[True, False]).ravel()
            
        data = {"nb_p": p, "nb_b": num_b, "tp": tp, "fn": fn, "fp": fp, "tn": tn}
        lst.append(data)

fig = plt.figure()
#plt.subplot(2,1,1)
plt.plot(a,"b+")
#plt.subplot(2,1,2)
plt.plot(b,"rx")
plt.show()
