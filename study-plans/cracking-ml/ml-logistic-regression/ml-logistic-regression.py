import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    x=np.asarray(X,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    w=np.zeros(x.shape[1],dtype=np.float64)
    b=0.0
    N=len(y)
    for _ in range(n_iters):
        z=x@w+b
        error=(1/(1+ np.exp(-z)))- y
        w-=lr*((x.T @ error)/N)
        b-= lr*((np.sum(error))/N)
    return w.tolist(),b
        
        
