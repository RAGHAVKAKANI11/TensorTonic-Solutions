import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    x=np.asarray(X,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    w=np.zeros(x.shape[1],dtype=np.float64)
    b=0.0
    N=len(y)
    for _ in range(epochs):
        error=x @ w + b - y
        dw= (2 / N) * (x . T @ error)+2*alpha*w
        db= (2 / N) * np.sum(error)
        w-=lr*dw
        b-=lr*db
    return w.tolist(),b
    
    
