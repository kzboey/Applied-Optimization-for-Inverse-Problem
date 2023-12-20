import numpy as np
from scipy.sparse import coo_matrix

"""
    forward difference    
"""
def forward_diff(N):
    row_indices = []
    col_indices = []
    data = []

    for i in range(N - 1):
        row_indices.append(i)
        col_indices.append(i)
        data.append(-1)
        row_indices.append(i)
        col_indices.append(i + 1)
        data.append(1)

    L = coo_matrix((data, (row_indices, col_indices)), shape=(N, N))

    return L

"""
    backward difference    
"""
def backward_diff(N):
    row_indices = []
    col_indices = []
    data = []
    
    for i in range(N - 1):
        row_indices.append(i+1)
        col_indices.append(i)
        data.append(-1)
        row_indices.append(i+1)
        col_indices.append(i + 1)
        data.append(1)
    
    L = coo_matrix((data, (row_indices, col_indices)), shape=(N, N))

    return L

"""
    central difference    
"""
def central_diff(N):
    row_indices = []
    col_indices = []
    data = []
    
    for i in range(N - 2):
        row_indices.append(i+1)
        col_indices.append(i)
        data.append(-0.5)
        row_indices.append(i+1)
        col_indices.append(i + 2)
        data.append(0.5)
    
    L = coo_matrix((data, (row_indices, col_indices)), shape=(N, N))

    return L