import math
import numpy as np

def var_wgt(X, W, method="nist"): 
    sumW = W.sum()

    if method == "nist": 
        # fixed.2009.03.07, divisor added. 
        xbarwt = mean_wgt(X, W)
        Np = (W != 0).sum()
        D = sumW * (Np - 1.0) / Np 

        return (W * (X - xbarwt)**2).sum() / D

    else: # default is R 
        sumW2 = (W**2).sum()
        xbarwt = mean_wgt(X, W)

        return (W * (x - xbarwt)**2).sum() * sumW / (sumW**2 - sumW2)

def mean_wgt(X, W):
    return sum(W * X) / W.sum()

def std_wgt(X, W, method="nist"):
    variance = var_wgt(X, W, method=method)
    return math.sqrt(variance)
