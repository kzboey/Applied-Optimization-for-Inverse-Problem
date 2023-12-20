import numpy as np

def getParams(args):
    v = None 
    x = None
    sigma = None
    beta = None
    delta = None
    proxG = None
    y_factor = None
    tau = None
    
    for key, value in args.items():
        if key == 'v':
            v = value
        elif key == 'x':
            x = value
        elif key == 'sigma': # step size, 1/L
            sigma = value
        elif key == 'beta': # regularization parameter of l2
            beta = value
        elif key == 'tau': # regularization parameter of l1
            tau = value
        elif key == 'delta': # huber parameter
            delta = value
        elif key == 'g':   # proximal operator of translation
            proxG = value
        elif key == 'y':   # translation factor y
            y_factor = value
    
    return v,x,sigma,beta,delta, proxG, y_factor,tau


def proximalIdentity(args):
    v,x,_,_,_,_,_,_ = getParams(args)
    return v

def proximalTranslation(args):
    v,x,sigma,beta,delta,g,y,tau = getParams(args)
    args['v'] = v-y
    return y + g(args)

## soft thresholding
def proximalL1(args):
    v,x,sigma,_,_,_,_,tau = getParams(args)
    threshold = sigma*tau
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.)

def proximalElasticNet(args):
    v,x,sigma,beta,_,_,_,tau = getParams(args)
    factor = 1 / (1 + sigma*beta)
    return factor*proximalL1(args)

def proximalL2Squared(args):
    v,x,sigma,beta,_,_,_,_ = getParams(args)
    factor = 1 / (1 + sigma*beta)
    return factor*v
    
def proximalHuber(args):
    v,x,sigma,_,delta,_,_,_ = getParams(args)
    maxTerm = max(np.linalg.norm(x),sigma)
    factor = 1 - sigma / (maxTerm + delta)
    return factor*v

def proximalL21(args):
    ## v is a matrix
    v,x,sigma,_,_,_,_,tau = getParams(args)
    lambd = sigma*tau
    factor = 1 - (lambd / (np.maximum(np.linalg.norm(v),lambd)))
    return factor*v