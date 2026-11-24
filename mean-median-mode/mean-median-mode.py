from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    x=np.asarray(x,dtype=float)
    x=np.sort(x)
    mean=np.mean(x)
    median=np.median(x)
    values, counts = np.unique(x, return_counts=True)
    modes=values[counts==np.max(counts)]
    mode=np.min(modes)
    return {
        "mean" : float(mean),
        "median" : float(median),
        "mode" : float(mode)
    }
    
    