import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gen_data import gen_data
from sklearn import linear_model as lin_mod
from sklearn.metrics import confusion_matrix

lst = []
n = 500
seed = 42
P = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
B = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
file = 'analyse_n500_bfixe_m0,9_l1,1.xlsx'


for p in P:
    for b in B:
        
        if p>b:
        
            num_b = b

            X, y, beta = gen_data(n, p, num_b, seed)

            reg = lin_mod.Lars()
            reg.fit(X, y)

            a = (reg.coef_>0.9)*(reg.coef_<1.1) #selection of actives 
            b = beta>0.9
            
            tp, fn, fp, tn = confusion_matrix(b,a, labels=[True, False]).ravel()
            
            data = {"nb_p": p, "nb_b": num_b, "tp": tp, "fn": fn, "fp": fp, "tn": tn}
            lst.append(data)
        
df = pd.DataFrame(lst)
df.to_excel(file, index=False)
'''
fig = plt.figure()
#plt.subplot(2,1,1)
plt.plot(a,"b+")
#plt.subplot(2,1,2)
plt.plot(b,"rx")
plt.show()

'''