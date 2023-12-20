import numpy as np
from scipy import sparse
from scipy.sparse import diags,spdiags

def power_iteration(A, num_iterations=1000):
    n, m = A.shape

    # Generate a random initial vector
    v = np.random.rand(m)

    # Run power iteration
    for i in range(num_iterations):
        w = A @ v
        v = w / np.linalg.norm(w)

    # Compute the largest singular value
    s = (v @ (A @ v) ) / np.linalg.norm(v)
    
    return s

def get_largest_sigma(A, num_iterations=1000):

    A_squared = (A.T).dot(A)
    
    eig_A_squared = power_iteration(A_squared, num_iterations)
    
    return np.sqrt(eig_A_squared)

# with finite difference
def get_regularizer_grad(regularizer,C,x,delta=1):
    grad = 0
    
    if regularizer == 'tikhonov':
        grad = ((C.T).dot(C)).dot(x)
    elif regularizer == 'huber':
        z = C.dot(x)
        abs_z = np.abs(z)
        grad = np.where(abs_z <= delta, z, delta * np.sign(z))
        grad = (C.T).dot(grad)
    elif regularizer == 'fair':
        z = C.dot(x)
        zTerm = np.abs(z/delta)
        grad = z / (1 + zTerm)
        grad = (C.T).dot(grad)
           
    return grad
        
# without finite difference
def get_regularizer_grad2(regularizer,x,delta=1):
    grad = 0
    
    if regularizer == 'tikhonov':
        grad = x
    elif regularizer == 'huber':
        z = x
        abs_z = np.abs(z)
        grad = np.where(abs_z <= delta, z, delta * np.sign(z))
    elif regularizer == 'fair':
        z = x
        zTerm = np.abs(z/delta)
        grad = z / (1 + zTerm)
           
    return grad
    
    
def compute_OGM1(A,b,L,x,C=None,eps=1e-6,beta=1,delta=1,regularizer='gd',iteration=1000):
    #x = np.zeros(A.shape[1])
    y = x.copy()
    theta = 1
    stopIdx = 0
    #L = 1000 GradientDescent.get_largest_sigma(A,100)
    if C is None:
        C = sparse.eye(A.shape[1])
    
    for i in range(iteration):
        grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        #grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad2(regularizer,x,delta)
        new_y = x - (1/L)*grad
        other_term = (1+np.sqrt(1+4*np.square(theta))) / 2
        last_term = (1+np.sqrt(1+8*np.square(theta))) / 2
        new_theta = np.where(i==iteration-1, last_term, other_term)
        new_x = new_y + ((theta-1)/new_theta)*(new_y-y)+(theta/new_theta)*(new_y-x)
        stopIdx=i
        if np.linalg.norm(new_x - x) < eps:
            break
        y = new_y
        theta = new_theta
        x = new_x

    print("final x: ",x)    
    return x, stopIdx+1 

def compute_GD(A,b,alpha,x,C=None,eps=1e-6,beta=1,delta=1,regularizer='gd',iteration=1000):
    #x = np.zeros(A.shape[1])
    stopIdx = 0
    if C is None:
        C = sparse.eye(A.shape[1])
    
    for i in range(iteration):
        #grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad2(regularizer,x,delta)
        new_x = x - alpha*grad
        stopIdx=i
        if np.linalg.norm(new_x-x) < eps:
            break
        x = new_x
    
    print("final x: ",x)
    return x, stopIdx+1

def compute_Landweber(A,b,lambd,x,eps=1e-6,beta=1,delta=1,iteration=1000):
    #largest singular value
    sigma = get_largest_sigma(A,1000)
    if lambd > sigma:
        lambd = format(sigma, '.0e')
    print('chosen w: ',lambd)
    stopIdx = 0
    #x = np.zeros(A.shape[1])
    
    for i in range(iteration):
        x_new = x - lambd*((A.T).dot(A.dot(x)-b))
        stopIdx=i
        if np.linalg.norm(x_new - x) < eps:
            break
        x = x_new
        
    return x, stopIdx+1


def conjugate_gradient_normal(C, d,x, eps=1e-6,iteration=1000):
    A = C.T.dot(C)
    b = C.T.dot(d)
    n = C.shape[1]
    #x = np.zeros(n)
    r = b-A.dot(x)
    p = r
    stopIdx = 0
    
    for i in range(iteration):
        alpha = r.T.dot(r) / p.T.dot(A.dot(p))
        x = x + alpha*p
        r_prev = r
        r = r - alpha * A.dot(p)
        stopIdx = i
        if np.linalg.norm(r-r_prev) < eps:
            break
        beta = r.T.dot(r) / r_prev.T.dot(r_prev)
        p = r + beta*p
        
    return x, stopIdx+1


def compute_FGM1(A,b,L,x,C=None,eps=1e-6,beta=1,delta=1,regularizer='gd',iteration=1000):
    #x = np.zeros(A.shape[1])
    y = x.copy()
    theta = 1
    stopIdx = 0
    #L = 1000 GradientDescent.get_largest_sigma(A,100)
    if C is None:
        C = sparse.eye(A.shape[1])
    
    for i in range(iteration):
        #grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad2(regularizer,x,delta)
        new_y = x - (1/L)*grad
        new_theta = (1+np.sqrt(1+4*np.square(theta))) / 2
        new_x = new_y + ((theta-1)/new_theta)*(new_y-y)
        stopIdx=i
        if np.linalg.norm(new_x - x) < eps:
            break
        y = new_y
        theta = new_theta
        x = new_x

    print("final x: ",x)    
    return x, stopIdx+1 

def compute_SIRT(A,b,lambd,x,eps=1e-6,iteration=1000):
    #largest singular value
    sigma = get_largest_sigma(A,1000)
    if lambd > sigma:
        lambd = format(sigma, '.0e')
    print('chosen w(sirt): ',lambd)
    stopIdx = 0
    cols = A.shape[1]
    rows = A.shape[0]
    D_vec = 1/(A.T.dot(np.ones(rows)))
    M_vec = 1/(A.dot(np.ones(cols)))
    D = spdiags(D_vec, 0, D_vec.size, D_vec.size)
    M = spdiags(M_vec, 0, M_vec.size, M_vec.size)
    AM = A.T @ M
    DAM =  D @ AM #(D.toarray()).dot(A.T.dot(M.toarray()))
    for i in range(iteration):
        x_new = x - lambd*DAM.dot((A.dot(x)-b))
        stopIdx=i
        if np.linalg.norm(x_new - x) < eps:
            break
        x = x_new
        
    return x, stopIdx+1