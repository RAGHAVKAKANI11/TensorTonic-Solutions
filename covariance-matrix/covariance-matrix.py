import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    x=np.asarray(X,dtype=float)
    z=np.mean(x,axis=0)
    N=len(x)
    x_c=x-z
    y=(x_c.T @ x_c)/(N-1)
    return y
    