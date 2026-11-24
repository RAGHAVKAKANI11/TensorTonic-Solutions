import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    a=np.asarray(A,dtype=float)
    a_t=a.T
    return a_t
