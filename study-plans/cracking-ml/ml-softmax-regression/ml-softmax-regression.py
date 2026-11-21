import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    x=np.asarray(X,dtype=float)
    y=np.asarray(y,dtype=int)
    n_features=x.shape[1]
    w=np.zeros((n_features,num_classes),dtype=float)
    b=np.zeros(num_classes,dtype=float)
    N=len(y)
    y_onehot=np.eye(num_classes)[y]
    for _ in range(n_iters):
        z=x@w+b
        z=z-np.sum(z,axis=1,keepdims=True)
        exp_z=np.exp(z)
        y_pred=exp_z/np.sum(exp_z,axis=1,keepdims=True)
        error=y_pred-y_onehot
        dw=(x.T@error)/N
        db=np.sum(error,axis=0)/N
        w-=lr*dw
        b-=lr*db
    return w.tolist(),b.tolist()
        
    
