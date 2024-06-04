import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gen_data import gen_data
from sklearn import linear_model as lin_mod
from sklearn.metrics import confusion_matrix

lst = []
n = 1000
seed = 42
P = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
percent_b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
file = 'analyse_n1000.xlsx'


for p in P:
    for per in percent_b:
        
        num_b = round(p*per)


        X, y, beta = gen_data(n, p, num_b, seed)

        reg = lin_mod.Lars()
        reg.fit(X, y)

        a = reg.coef_>0 #selection of actives 
        b = beta>0
        
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