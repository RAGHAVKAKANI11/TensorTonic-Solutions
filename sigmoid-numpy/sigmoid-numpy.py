import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    x=np.asarray(x,dtype=float)
    sigma=1/(1+np.exp(-x))
    if sigma.ndim==0:
        return float(sigma)
    return sigma