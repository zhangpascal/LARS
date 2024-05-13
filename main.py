import numpy as np

def pre_proc(X,y):
    X = np.array(X)
    y = np.array(y)
    
    X = X-np.mean(X, axis=0)
    X = X/np.linalg.norm(X, axis=0)
    y = y-np.mean(y)
    
    return X, y

def my_lars(X,y):
    
    n = len(X)
    p = len(X[0])
      
    X_active = np.array([[] for i in range(n)])
    
    mu = np.zeros(n)
    
    for i in range(p):
        
        c = np.dot(X.T,y-mu)
        
        c_max_ind = np.argmax(np.absolute(c))
        c_sign = np.sign(c[c_max_ind])
        c_max = np.absolute(c[c_max_ind])

        X_active = np.append(X_active, c_sign*X[:,c_max_ind].reshape(-1,1), axis=1)
        X = np.delete(X, c_max_ind, 1)
        
        g_inv=np.linalg.inv(np.dot(X_active.T,X_active))

        a_max =1/np.sqrt(np.sum(g_inv))
        
        u = np.dot(X_active,np.sum(a_max*g_inv,axis = 1))
        
        c_inactive =  np.delete(c,c_max_ind)
        a_inactive = np.dot(X.T, u)
        
        if i < p-1:
            gamma1 = (c_max - c_inactive)/(a_max - a_inactive)
            gamma2 = (c_max + c_inactive)/(a_max + a_inactive)
            gamma = np.min([gamma1[gamma1>=0], gamma2[gamma2>=0]])
        else:
            gamma = c_max/a_max
        
        mu = mu + gamma*u
        
    return mu

X, y = pre_proc(X, y)

mu = my_lars(X,y)
beta = np.dot(np.linalg.inv(X[:len(X[0]),:]),mu[:len(X[0])])

print(beta)
