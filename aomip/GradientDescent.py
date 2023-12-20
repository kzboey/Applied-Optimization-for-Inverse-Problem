import numpy as np
from scipy import sparse
from scipy.sparse import diags,spdiags
import aomip
import time

eps = 1e-6

# Get regularization term of objective function
def getFuncReg(regularizer,x,delta=1):
    regTerm = 0
    
    if regularizer == 'tikhonov':
        regTerm = 0.5*np.linalg.norm(x)**2
    elif regularizer == 'huber':
        regTerm =  np.where(x <= delta, 0.5*np.square(x), 0.5*delta * (np.abs(x) - 0.5*delta ))
    elif regularizer == 'fair':
        z = np.abs(x/delta)
        regTerm = np.square(delta)*z - np.log(1 + z)
    elif regularizer == 'lasso':
        regTerm = np.linalg.norm(x,1)
        
    return regTerm

# Get objective function value
def getFunc(A,b,x,beta,delta,regularizer,tao=0):
    residual = A.dot(x) - b
    if regularizer == 'elastic':
        return 0.5 * np.linalg.norm(residual)**2 + tao*getFuncReg('lasso',x,delta) + beta*getFuncReg('tikhonov',x,delta)
    else:
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

# to compute largest singular value of matrix A
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

"""
    Optimized Gradient method 1
"""
def compute_OGM1(A,b,x,C=None,beta=0,delta=0,tau=0,regularizer='gd',mygrad=None,iteration=1000,restart=False,restartItr=100,callback=None):
    #x = np.zeros(A.shape[1])
    y = x.copy()
    theta = 1
    stopIdx = 0
    L = get_largest_sigma(A,1000)**2
    alpha = float(format(1/L, '.0e'))
    print('step size OGM: ',alpha)
    if C is None:
        C = sparse.eye(A.shape[1])
    ## Restart condition initialization
    restart_counter = 0
    theta_initial = theta
    
    for i in range(iteration):
        if mygrad is None:
            grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        else:
            grad = mygrad(x)+ beta*get_regularizer_grad(regularizer,C,x,delta)
        
        new_y = x - alpha*grad
        other_term = (1+np.sqrt(1+4*np.square(theta))) / 2
        last_term = (1+np.sqrt(1+8*np.square(theta))) / 2
        new_theta = np.where(i==iteration-1, last_term, other_term)
        new_x = new_y + ((theta-1)/new_theta)*(new_y-y)+(theta/new_theta)*(new_y-x)

        if np.linalg.norm(new_x - x) < eps:
            break
        y = new_y
        theta = new_theta
        x = new_x
        stopIdx=i
        
        if callback:
            objValue = getFunc(A,b,x,beta,delta,regularizer,tau)
            callback(np.copy(x),np.copy(objValue), i)
            
        if restart and i%restartItr == 0:
            theta=theta_initial
        restart_counter+=1
        
    print("final x: ",x)    
    return x, stopIdx+1

"""
    Gradient Descent
"""
def compute_GD(A,b,alpha,x,C=None,beta=0,delta=0,regularizer='gd',iteration=1000):
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

"""
    Landweber iteration
"""
def compute_Landweber(A,b,lambd,x,beta=0,delta=0,iteration=1000):
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
    
    print("final x: ",x) 
    return x, stopIdx+1, history


def conjugate_gradient_normal(C, d,x,iteration=1000):
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
    
    print("final x: ",x) 
    return x, stopIdx+1, history 

# remove L, homework 4
def compute_FGM1(A,b,x,C=None,beta=0,delta=0,regularizer='gd',iteration=1000):
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
    
    print("final x: ",x) 
    return x, stopIdx+1, history 

def compute_SIRT(A,b,lambd,x,iteration=1000):
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
    
    print("final x: ",x) 
    return x, stopIdx+1, history

"""
    alpha: initial step size
    con1: control parameter 1, p ∈ (0, 1)
    con2: control parameter 2, c ∈ (0, 1)
    GD with backtracking
"""
def backtracking_linesearch(A,b,x,alpha=1e-3, con1=0.5,con2=0.5,C=None,beta=0,delta=0,regularizer='gd',iteration=1000,restart=False,restartItr=100,callback=None):
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
    
    print("final x: ",x) 
    return x, stopIdx+1

"""
    Barzilai and Borwein method
    bb: 1 (BB1), 2 (BB2)
"""
def bb_linesearch(A,b,x,alpha=1e-3,C=None,beta=0,delta=0,tau=0,bb=1,regularizer='gd',mygrad=None,iteration=1000,restart=False,restartItr=50,callback=None):
    stopIdx = 0
    if C is None:
        C = sparse.eye(A.shape[1])

    p_prev = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
    x_prev = x 
    
    ## Restart condition initialization
    restart_counter = 0
    alpha_initial = alpha
    
    start_time = time.time() 
    for i in range(iteration):
        x = x - alpha*p_prev
        p = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        if mygrad is None:
            p = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        else:
            p = mygrad(x) + beta*get_regularizer_grad(regularizer,C,x,delta)
        y = p - p_prev
        p_prev = p      
        s = x - x_prev
        x_prev = x
        
        if bb == 1:
            alpha = s.T.dot(y) / (y.T).dot(y)
        else:
            alpha = s.T.dot(s) / s.T.dot(y)
        
        if np.linalg.norm(s) < eps:
            break
        stopIdx = i
        
        if restart and i%restartItr == 0:
            alpha = alpha_initial
        restart_counter+=1
        
        if callback:
            objValue = getFunc(A,b,x,beta,delta,regularizer,tau)
            callback(np.copy(x),np.copy(objValue), i)
    
    elapsed_time = time.time() - start_time

    print("Elapsed time:", elapsed_time, "seconds")     
    return x, stopIdx+1
    
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

def ista(A,b,x,beta=0,mygrad=None,iteration=1000, backtrack=False):
    stopIdx = 0
    # history = np.zeros((x.size, iteration+1))
    # history[:, 0] = x
    L = get_largest_sigma(A,1000)**2 # Lipschitz constant
    alpha = 1/L
    
    start_time = time.time() 
    for i in range(iteration):
        if mygrad is None:
            grad = (A.T).dot(A.dot(x)- b)
        else:
            grad = mygrad(x)
        if backtrack:
            alpha = backtracking(A,b,x,alpha)
        #print('alpha: ',alpha)
        x_new = soft_thresh(x - alpha*grad, beta*alpha)
        if np.linalg.norm(x_new-x) < eps:
            break
        stopIdx = i
        x = x_new
        # history[:,i+1] = x
    
    elapsed_time = time.time() - start_time

    print("Elapsed time:", elapsed_time, "seconds")
    return x, stopIdx+1

"""
  Projected Gradient Descent
"""
def project_onto_box(x, lower_bound, upper_bound):
    projected_x = np.minimum(np.maximum(x,lower_bound),upper_bound)
    return projected_x
  
def projected_gradient_descent(A,b,x,alpha,lower_bound, upper_bound, C=None, beta=1,delta=1,regularizer='gd',iteration=1000,callback=None):
    stopIdx = 0
    if C is None:
        C = sparse.eye(A.shape[1])
    ## Restart condition initialization
    restart_counter = 0
    x_initial = x.copy()
        
    for i in range(iteration):
        grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        x_new = x - alpha * grad
        x_new = project_onto_box(x_new,lower_bound,upper_bound)
        if np.linalg.norm(x_new-x) < eps:
            break
        stopIdx = i
        x = x_new
        if callback:
            objValue = getFunc(A,b,x,beta,delta,regularizer,0)
            callback(np.copy(x),np.copy(objValue), i)
     
    print("final x: ",x) 
    return x, stopIdx+1

"""
  Proximal Gradient Method
  apply projection with arbitrary proximal operator
"""
def pgm(A,b,x,proxOperator,proxParams,beta=0,delta=0,regularizer='gd',iteration=1000,callback=None):
    x_initial = x.copy()
    stopIdx = 0
    L = get_largest_sigma(A,1000)**2 # Lipschitz constant
    alpha = 1/L
    alpha = float(format(1/L, '.0e'))
    proxParams['sigma'] = alpha
    C = sparse.eye(A.shape[1])
    tau = 0
    if 'tau' in proxParams:
        tau = proxParams['tau']
        
    ## Restart condition initialization
    restart_counter = 0
    x_initial = x.copy()
    
    start_time = time.time() 
    for i in range(iteration):
        grad = (A.T).dot(A.dot(x)- b) + beta*get_regularizer_grad(regularizer,C,x,delta)
        v = x - alpha*grad
        proxParams['x'] = x
        proxParams['v'] = v
        x_new = proxOperator(proxParams)
        if np.linalg.norm(x_new-x) < eps:
            break
        stopIdx = i
        x = x_new
        if callback:
            objValue = getFunc(A,b,x,beta,delta,regularizer,tau)
            callback(np.copy(x),np.copy(objValue), i)

    elapsed_time = time.time() - start_time

    print("Elapsed time:", elapsed_time, "seconds")        
    return x, stopIdx+1
    
    
"""
  Fast Proximal Gradient Method
"""    
def fast_pgm(A,b,x,proxOperator,proxParams,beta=0,delta=0,mygrad=None,regularizer='gd',momentum=1,iteration=1000,restart=False,restartItr=100,callback=None):
    stopIdx = 0
    L = get_largest_sigma(A,1000)**2 # Lipschitz constant
    alpha = 1/L
    alpha = float(format(1/L, '.0e'))
    proxParams['sigma'] = alpha
    C = sparse.eye(A.shape[1])
    z = x.copy()
    alphaM_prev = 0
    tM_prev = 1
    tM = 0
    tau = 0
    if 'tau' in proxParams:
        tau = proxParams['tau']
        
    ## Restart condition initialization
    restart_counter = 0
    tm_initial = 1
    alphaM_initial = 0
    x_initial = x
    
    start_time = time.time() 
    for i in range(iteration):
        if mygrad is None:
            grad = (A.T).dot(A.dot(x)- b) + beta*get_regularizer_grad(regularizer,C,x,delta)
        else:
            grad = mygrad(x) + beta*get_regularizer_grad(regularizer,C,x,delta)
            
        v = x - alpha*grad
        proxParams['x'] = x
        proxParams['v'] = v
        z_new = proxOperator(proxParams)
        if momentum == 1:
            alphaM = (i-1)/(i+2)
        else:
            tM = (1 + np.sqrt(1 + 4 * tM_prev**2)) / 2
            alphaM =  (tM_prev-1)/tM
        x_new =  z_new + alphaM*(z_new - z)
        if np.linalg.norm(x_new-x) < eps:
            break
        stopIdx = i
        alphaM_prev = alphaM
        tM_prev = tM
        x = x_new
        z = z_new
        if callback:
            objValue = getFunc(A,b,x,beta,delta,regularizer,tau)
            callback(np.copy(x),np.copy(objValue), i)
            
        if restart and i%restartItr == 0:
            tM = tm_initial
            alphaM_prev = alphaM_initial
            tM_prev = tm_initial  
            x = x_initial
        restart_counter+=1
    
    elapsed_time = time.time() - start_time

    print("Elapsed time:", elapsed_time, "seconds")  
    return x, stopIdx+1
 
"""
    Proximal Optimized Gradient Method
"""
def pogm(A,b,x,proxOperator,proxParams,C=None,beta=0,delta=0,tau=0,mygrad=None,regularizer='gd',iteration=1000,restart=False,restartItr=100,callback=None):
    stopIdx = 0
    L = get_largest_sigma(A,1000)**2 # Lipschitz constant
    alpha = 1/L
    alpha = float(format(1/L, '.0e'))
    theta = 1
    gamma = 1
    theta_initial = theta
    w = x.copy()
    z = x.copy()
    tau = 0
    if 'tau' in proxParams:
        tau = proxParams['tau']
        
    ## Restart condition initialization
    restart_counter = 0
    theta_initial = theta
    gamma_initial = gamma
    w_initial = w
    z_intial = z
    x_initial = x
        
    start_time = time.time() 
    for i in range(iteration):
        if mygrad is None:
            grad = (A.T).dot(A.dot(x)-b)+beta*get_regularizer_grad(regularizer,C,x,delta)
        else:
            grad = mygrad(x) + beta*get_regularizer_grad(regularizer,C,x,delta)
            
        other_term = (1+np.sqrt(1+4*np.square(theta))) / 2
        last_term = (1+np.sqrt(1+8*np.square(theta))) / 2
        new_theta = np.where(i==iteration-1, last_term, other_term)
        
        new_gamma = alpha * ((2*theta + new_theta -1) / new_theta)
        new_w = x - alpha * grad      
        new_z = new_w + ((theta-1)/new_theta)*(new_w - w) + (theta/new_theta)*(new_w-x) + (alpha * (theta-1) / (gamma*new_theta))*(z-x)
        
        proxParams['sigma'] = new_gamma
        proxParams['x'] = x
        proxParams['v'] = new_z
        new_x = proxOperator(proxParams)
        
        if np.linalg.norm(new_x - x) < eps:
            break
        theta = new_theta
        gamma = new_gamma
        x = new_x
        w = new_w
        z = new_z
        stopIdx=i
        
        if callback:
            objValue = getFunc(A,b,x,beta,delta,regularizer,tau)
            callback(np.copy(x),np.copy(objValue), i)
            
        if restart and i%restartItr == 0:
            theta = theta_initial
            gamma = gamma_initial
            w = w_initial
            z = z_intial
            x = x_initial
        restart_counter+=1
    
    elapsed_time = time.time() - start_time
    print("Elapsed time:", elapsed_time, "seconds")          
    return x, stopIdx+1

"""
     Linearized ADMM
"""
def admm_lasso(A,b,x,proxf,proxg,proxParamsf,proxParamsg,tau=0,C=None,iteration=1000,callback=None):
    stopIdx = 0
    A_T = A.T
    z = A.dot(x)
    u = np.zeros_like(b)# dual variable
    L = get_largest_sigma(A,100)**2 # norm of K
    alpha = 1/L
    alpha = float(format(1/L, '.0e'))
    proxParamsf['sigma'] = alpha
    proxParamsg['sigma'] = alpha
    lambd = (0.95*tau)/ (L)
    proxParamsf['tau']  = lambd
    proxParamsg['beta']  = 1

    print("tau: ",tau)   
    for i in range(iteration):
        x_old = x
        
        # update primer
        v = x - ((lambd/tau)*A_T.dot(A.dot(x) - z + u))
        proxParamsf['v'] = v
        x = proxf(proxParamsf)
        
        v = A.dot(x) + u  
        proxParamsg['v'] = v
        
        z = proxg(proxParamsg)
        
        # update dual
        u = u + A.dot(x) - z
      
        if np.linalg.norm(x - x_old) < eps:
            break
        
        stopIdx=i
              
        if callback:
            objValue = getFunc(A,b,x,beta,delta,regularizer,tau)
            callback(np.copy(x),np.copy(objValue), i)
    elapsed_time = time.time() - start_time

    print("Elapsed time:", elapsed_time, "seconds")    
    return x, stopIdx+1

"""
     Linearized ADMM for CT: use foward and anjoint operator instead of matrix operation
"""
def admm_lasso_ct(A,b,x,proxf,proxg,proxParamsf,proxParamsg,lambd=0,tau=0,iteration=1000,callback=None):
    stopIdx = 0
    z = A.apply(x)
    u = np.zeros_like(b) # dual variable

    start_time = time.time()  
    for i in range(iteration):
        x_old = x
        
        # update primer
        v = x - ((lambd/tau)*A.applyAdjoint(A.apply(x) - z + u))
        proxParamsf['v'] = v
        x = proxf(proxParamsf)
         
        v = A.apply(x) + u  
        proxParamsg['v'] = v
        
        z = proxg(proxParamsg)
        
        # update dual
        u = u + A.apply(x) - z
      
        if i>100 and np.linalg.norm(x - x_old) < eps:
            break
        
        stopIdx=i
        
        if callback:
            callback(np.copy(x), i)
    elapsed_time = time.time() - start_time

    print("Elapsed time:", elapsed_time, "seconds")
        
    return x, stopIdx+1

"""
    for 2D image only
    A: forward operator
    b: sinogram
    x: current point (stacked image)
    grad: gradient operator
    tau: TV regularization parameter
    proxg: proximal parameter for function g (block)
    proxParamsG: proximal parameter for proximal operator G (block)
    
"""
def admm_tv(b,x,stackedOpt,opts,proxG,proxParamsG,lambd=0,tau=0,iteration=1000,callback=None):
    
    stopIdx = 0
    z = stackedOpt.apply(opts,x) # stacked
    u1 = np.zeros_like(b[0])
    u2 = np.zeros_like(b[1])
    u = [u1,u2]
    img = x[0]
    scalar = lambd/tau
    
    start_time = time.time()
    for i in range(iteration):
        img_old = img
        zu = stackedOpt.addition(z,u)
        adjTerm = stackedOpt.subtraction(stackedOpt.apply(opts,x), zu) 

        x = stackedOpt.subtraction(x, stackedOpt.scalar_mut(stackedOpt.applyAdjoint(opts,adjTerm), scalar))

        v = stackedOpt.addition(stackedOpt.apply(opts,x), u)

        blockZ = []
        
        for count, (prox,proxParam) in enumerate(zip(proxG,proxParamsG)):
            proxParam['v'] = v[count]
            blockZ.append(prox(proxParam))
            
        z = blockZ
        
        u = stackedOpt.addition(u, stackedOpt.subtraction(stackedOpt.apply(opts,x), z))
        
        img = x[0]
        
        if i>100 and np.linalg.norm(img - img_old) < eps:
            break
        
        stopIdx=i
        
        if callback:
            callback(np.copy(img), i)
    elapsed_time = time.time() - start_time

    print("Elapsed time:", elapsed_time, "seconds")
        
    return img, stopIdx+1
        
        
"""
    subgradient method with TV regularization
"""        
def subgradient(A,b,x,beta,alpha,mygrad,iteration=1000,callback=None):
    grad = aomip.FirstDerivative()
    stopIdx = 0
    
    start_time = time.time()
    for i in range(iteration):
        subgrad = mygrad(x) + beta*np.sign(np.sum(grad.apply(x),axis=0))
        new_x = x - alpha[i]*subgrad
        stopIdx=i
        if np.linalg.norm(new_x-x) < 1e-6:
            break
        x = new_x

        if callback:
            callback(np.copy(x), i)
            
    elapsed_time = time.time() - start_time

    print("Elapsed time:", elapsed_time, "seconds")
     
    return x, stopIdx+1
    
    
    
    
    
    