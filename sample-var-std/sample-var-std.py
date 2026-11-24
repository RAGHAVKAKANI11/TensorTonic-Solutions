import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x=np.asarray(x,dtype=float)
    N=len(x)
    x_mean = np.mean(x)
    x_variance=np.sum((x-x_mean)**2)/(N-1)
    x_standard_deviation=np.sqrt(x_variance)
    return {
        "variance": float(x_variance),
        "standard_deviation": float(x_standard_deviation)
    }
    