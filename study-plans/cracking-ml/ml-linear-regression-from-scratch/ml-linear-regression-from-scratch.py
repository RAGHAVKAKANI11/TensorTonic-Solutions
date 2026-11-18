import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
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
        w -= lr * (2/N) * ( x.T @ error)
        b -= lr * (2/N) * np.sum(error)
    return w.tolist(),b
        