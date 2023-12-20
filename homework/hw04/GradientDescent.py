import numpy as np
from scipy import sparse
from scipy.sparse import diags,spdiags

def getFuncReg(regularizer,x,delta=1):
    regTerm = 0
    
    if regularizer == 'tikhonov':
        regTerm = np.linalg.norm(x)**2
    # elif regularizer == 'huber':
    #     regTerm = x**2
    # elif regularizer == 'fair':
    #     regTerm = x**2
    elif regularizer == 'lasso':
        regTerm = np.linalg.norm(x,1)
        
    return regTerm
    
def getFunc(A,b,x,beta,delta,regularizer):
    residual = A.dot(x) - b
    return 0.5 * np.linalg.norm(residual)**2 + beta*getFuncReg(regularizer,x,delta)

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
    elif regularizer == 'fair':
        z = C.dot(x)
        zTerm = np.abs(z/delta)
        grad = z / (1 + zTerm)
           
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
    
# remove L, homework 4
def compute_OGM1(A,b,x,C=None,eps=1e-6,beta=1,delta=1,regularizer='gd',iteration=1000):
    #x = np.zeros(A.shape[1])
    y = x.copy()
    theta = 1
    stopIdx = 0
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
    L = get_largest_sigma(A,1000)**2
    alpha = format(1/L, '.0e')
    print('step size OGM: ',alpha)
    if C is None:
        C = sparse.eye(A.shape[1])
    
    for i in range(iteration):
        grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
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
        history[:,i+1] = x
        
    print("final x: ",x)    
    return x, stopIdx+1, history

def compute_GD(A,b,alpha,x,C=None,eps=1e-6,beta=1,delta=1,regularizer='gd',iteration=1000):
    #x = np.zeros(A.shape[1])
    stopIdx = 0
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
    if C is None:
        C = sparse.eye(A.shape[1])
    
    for i in range(iteration):
        grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        #grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad2(regularizer,x,delta)
        new_x = x - alpha*grad
        stopIdx=i
        if np.linalg.norm(new_x-x) < eps:
            break
        x = new_x
        history[:,i+1] = x
        
    print("final x: ",x)
    return x, stopIdx+1, history

def compute_Landweber(A,b,lambd,x,eps=1e-6,beta=1,delta=1,iteration=1000):
    #largest singular value
    sigma = get_largest_sigma(A,500)
    upper_bound = 2/(sigma**2)
    if lambd > upper_bound:
        lambd = format(upper_bound, '.0e')
    print('chosen w: ',lambd)
    stopIdx = 0
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
    
    for i in range(iteration):
        x_new = x - lambd*((A.T).dot(A.dot(x)-b))
        stopIdx=i
        if np.linalg.norm(x_new - x) < eps:
            break
        x = x_new
        history[:,i+1] = x
        
    return x, stopIdx+1, history


def conjugate_gradient_normal(C, d,x, eps=1e-6,iteration=1000):
    A = C.T.dot(C)
    b = C.T.dot(d)
    n = C.shape[1]
    #x = np.zeros(n)
    r = b-A.dot(x)
    p = r
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
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
        history[:,i+1] = x
        
    return x, stopIdx+1, history 

# remove L, homework 4
def compute_FGM1(A,b,x,C=None,eps=1e-6,beta=1,delta=1,regularizer='gd',iteration=1000):
    #x = np.zeros(A.shape[1])
    y = x.copy()
    theta = 1
    stopIdx = 0
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
    L = get_largest_sigma(A,1000)**2 
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
        history[:,i+1] = x
        
    return x, stopIdx+1, history 

def compute_SIRT(A,b,lambd,x,eps=1e-6,iteration=1000):
    #largest singular value
    sigma = get_largest_sigma(A,1000)
    upper_bound = 2/(sigma**2)
    if lambd > upper_bound:
        lambd = format(upper_bound, '.0e')
    print('chosen w(sirt): ',lambd)
    stopIdx = 0
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
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
        history[:,i+1] = x
        
    return x, stopIdx+1, history

"""
    alpha: initial step size
    con1: control parameter 1, p ∈ (0, 1)
    con2: control parameter 2, c ∈ (0, 1)
    GD with backtracking
"""
def backtracking_linesearch(A,b,x,alpha=1e-3,con1=0.5,con2=0.5,C=None,eps=1e-6,beta=1,delta=1,regularizer='gd',iteration=1000,plot=False):
    stopIdx = 0
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
    if C is None:
        C = sparse.eye(A.shape[1])
    
    for i in range(iteration):
        grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        grad_norm = np.linalg.norm(grad)
        alphaj = alpha
        # armijio condition
        while getFunc(A,b,x + alphaj * grad,beta,delta,regularizer) <= getFunc(A,b,x,beta,delta,regularizer) + con2 * alphaj * grad_norm**2:
            alphaj = con1 * alphaj
        
        x_new = x - alphaj*grad
        stopIdx=i
        if np.linalg.norm(x_new-x) < eps:
            break
        x = x_new
        history[:,i+1] = x
        
    return x, stopIdx+1, history

"""
    Barzilai and Borwein method
    bb: 1 (BB1), 2 (BB2)
"""
def bb_linesearch(A,b,x,alpha=1e-3,C=None,eps=1e-6,beta=1,delta=1,bb=1,regularizer='gd',iteration=1000):
    stopIdx = 0
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
    if C is None:
        C = sparse.eye(A.shape[1])
    p_prev = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
    x_prev = x 
    
    for i in range(iteration):
        x = x - alpha*p_prev
        p = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        y = p - p_prev
        p_prev = p      
        s = x - x_prev
        x_prev = x
        
        if bb == 1:
            alpha = s.T.dot(y) / (y.T).dot(y)
        else:
            alpha = s.T.dot(s) / s.T.dot(y)
        
        stopIdx = i
        if np.linalg.norm(s) < eps:
            break
        history[:,i+1] = x
        
    return x, stopIdx+1, history
    
"""
    for ISTA used only (for now!!)
    alpha: initial step size
    con1: control parameter 1, p ∈ (0, 1)
    con2: control parameter 2, c ∈ (0, 1)
"""
def backtracking(A,b,x,alpha,con1=0.5,con2=0.5,beta=1,delta=1,regularizer='gd'):
    
    grad = (A.T).dot(A.dot(x)-b)
    grad_norm = np.linalg.norm(grad)
    alphaj = alpha
        
    # armijio condition
    while getFunc(A,b,x + alphaj * grad,0,0,'gd') <= getFunc(A,b,x,0,0,'gd') + con2 * alphaj * grad_norm**2:
        alphaj = con1 * alphaj
     
    return alphaj

"""
  ISTA
"""

def soft_thresh(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.)

def ista(A,b,x,beta=1,eps=1e-6,iteration=1000, backtrack=False):
    stopIdx = 0
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
    L = get_largest_sigma(A,1000)**2 # Lipschitz constant
    alpha = 1/L
    
    for i in range(iteration):
        grad = (A.T).dot(A.dot(x)-b)
        if backtrack:
            alpha = backtracking(A,b,x,alpha)
        #print('alpha: ',alpha)
        x_new = soft_thresh(x - alpha*grad, beta*alpha)
        if np.linalg.norm(x_new-x) < eps:
            break
        stopIdx = i
        x = x_new
        history[:,i+1] = x
        
    return x, stopIdx+1, history

"""
  Projected Gradient Descent
"""
def project_onto_box(x, lower_bound, upper_bound):
    projected_x = np.minimum(np.maximum(x,lower_bound),upper_bound)
    return projected_x
  
def projected_gradient_descent(A,b,x,alpha,lower_bound, upper_bound, C=None, beta=1,delta=1,regularizer='gd',eps=1e-6,iteration=1000):
    stopIdx = 0
    history = np.zeros((x.size, iteration+1))
    history[:, 0] = x
    if C is None:
        C = sparse.eye(A.shape[1])
        
    for i in range(iteration):
        grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        x_new = x - alpha * grad
        x_new = project_onto_box(x_new,lower_bound,upper_bound)
        if np.linalg.norm(x_new-x) < eps:
            break
        stopIdx = i
        x = x_new
        history[:,i+1] = x
        
    return x, stopIdx+1, history
