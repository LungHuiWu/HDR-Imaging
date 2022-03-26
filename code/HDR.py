### Arguments
# Z[i,j] = Value of pixel i on image j
# B[j] = Log delta t of image j
# l = lambda, constant for smoothness
# w[z] = Weighting function for each pixel value z

### Returns
# g[z] = log exposure for each pixel value z
# lE[i] = log film irradiance at pixel location i

import numpy as np
import sys
import cv2
from scipy.linalg import svd
from numpy.linalg import pinv

def solve_svd(A,b):
    
    invA = pinv(A)
    x = np.dot(invA,b)
    
    return x

def gsolve(Z,B,l,w):
    
    num_pixel = Z.shape[0]
    num_image = Z.shape[1]
    
    n = 256
    A = np.zeros((num_pixel*num_image+1+(n-2), n+num_pixel))
    b = np.zeros((A.shape[0], 1))
    
    k = 0
    for i in range(num_pixel):
        for j in range(num_image):
            wij = w[Z[i,j]]
            A[k,Z[i,j]] = wij
            A[k,n+i] = -wij
            # print(Z[i,j], k, wij, i, j, np.nonzero(A[k]))
            b[k] = wij * B[j]
            k += 1
            
    A[k,127] = 1
    k += 1
    
    for i in range(n-2):
        wi = l * w[i+1]
        A[k,i], A[k,i+1], A[k,i+2] = wi,-2*wi,wi
        k += 1
    
    x = solve_svd(A,b)
    g, lE = x[:n], x[n:]
    
    # for i in range(len(lE)):
    #     wi = np.array([w[j] for j in Z[i]])
    #     lE[i] = (wi*(Z[i]-B)).sum() / wi.sum()
    
    return g, lE