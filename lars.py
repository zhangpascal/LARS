import numpy as np
import time

def pre_proc(X,y):
    X = np.array(X)
    y = np.array(y)
    
    X = X-np.mean(X, axis=0)
    X = X/np.linalg.norm(X, axis=0)
    y = y-np.mean(y)
    
    return X, y

def min_plus(M, N):
    k=0
    sign=-1
    l=len(M)
    min_p = N[0]
    for i in range(1,l):
        if (N[i]<min_p) and (N[i]>=0):
            min_p = N[i]
            k=i
    for i in range(l):
        if (M[i]<min_p) and (M[i]>=0):
            min_p = M[i]
            k=i
            sign=1
    return min_p, k, sign

class LARS:
    def _init_(self):
        self.mu = np.array([])
        self.beta = np.array([])
        self.coef = np.array([])
        self.time = 0
        
    def fit(self, X, y):
        
        t1=time.time()
        
        n = len(X)
        p = len(X[0])
        
        self.mu = np.zeros(n)
        self.beta = np.zeros((p+1,p))
        
        mask_active = np.zeros(p, bool)
        mask_sign = np.zeros(p)
        lst_active = np.array([],dtype=int)
        
        X_active = np.array([[] for i in range(n)])
        X_inactive = X
        
        c = np.dot(X_inactive.T,y-self.mu)
        c_inactive = c
        c_max_ind = np.argmax(np.absolute(c))
        c_max = np.absolute(c[c_max_ind])
        c_sign = np.sign(c[c_max_ind])
        
        for i in range(p):

            max_index = np.where(mask_active==False)[0][c_max_ind]
            mask_sign[max_index] = c_sign
            lst_active = np.append(lst_active, max_index)
            mask_active[max_index]=True

            X_active = np.append(X_active,c_sign*X[:,max_index].reshape(-1,1), axis=1)

            X_inactive = np.delete(X_inactive, c_max_ind, 1)

            g_inv=np.linalg.inv(np.dot(X_active.T,X_active))

            a_max =1/np.sqrt(np.sum(g_inv))
            w = np.sum(a_max*g_inv,axis = 1)

            u = np.dot(X_active,w)

            c_inactive = np.delete(c_inactive,c_max_ind)
            a_inactive = np.dot(X_inactive.T, u)

            if i < p-1:
                gamma1 = (c_max - c_inactive)/(a_max - a_inactive)
                gamma2 = (c_max + c_inactive)/(a_max + a_inactive)
                gamma, c_max_ind, c_sign = min_plus(gamma1, gamma2)
            else:
                gamma = c_max/a_max
                
            c_max = c_max - gamma*a_max
            
            c_inactive = c_inactive - gamma*a_inactive
            
            self.mu = self.mu + gamma*u
            
            coef_step = gamma*w
            
            gamma_w = np.zeros(p)
            for j in range(len(lst_active)):
                gamma_w[abs(lst_active[j])] = coef_step[j]

            self.beta[i+1] = self.beta[i] + gamma_w
        
        self.beta = self.beta*mask_sign
        
        self.coef = self.beta[-1]
        
        t2=time.time()
        self.time=t2-t1
        
