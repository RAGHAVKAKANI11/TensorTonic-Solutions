import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float)
    x= np.dot(a,b)
    a=np.linalg.norm(a)
    b=np.linalg.norm(b)
    if a==0 or b==0 :
        return 0.0
    else:
        z=x/(a*b)
        return float(z)