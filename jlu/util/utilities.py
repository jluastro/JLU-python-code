import numpy as np

def triu2flat(m):
    """ 
    Flattens an upper triangular matrix, returning a vector of the
    non-zero elements. 
    """
    dim = m.shape[0]
    res = np.zeros(dim * (dim + 1) / 2)
    index = 0
    for row in range(dim):
        res[index:index + dim - row] = m[row, row:]
        index += dim - row

    return res

def flat2triu(a, dim):
    """ 
    Produces an upper triangular matrix of dimension dim from the 
    elements of the given vector. 
    """
    res = np.zeros((dim, dim))
    index = 0
    for row in range(dim):
        res[row, row:] = a[index:index + dim - row]
        index += dim - row

    return res
  
 
