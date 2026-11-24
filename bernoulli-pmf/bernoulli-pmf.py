import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    # Write code here
    x=np.asarray(x,dtype=float)
    p=np.asarray(p,dtype=float)
    pmf=np.zeros(len(x),dtype=float)
    pmf[x==0]=1-p
    pmf[x==1]=p
    variance=p*(1-p)
    mean=p
    return {
        "pmf": pmf,
        "mean": float(mean),
        "variance": float(variance)
    }
        
    
        

    