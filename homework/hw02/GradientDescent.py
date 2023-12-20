import sympy
import numpy as np

def df(f, vars):
    return sympy.Matrix([sympy.diff(f, var) for var in vars])

def gradient_descent(f, vars, x0, lr=0.1, eps=1e-6, max_iter=1000):
    ndim = x0.size
    x_cur = x0
    x_hist = [x0]
    stpIdx=0;
    
    for i in range(max_iter):
        grad = df(f, vars)
        if ndim == 1:
            grad_value = np.array(grad.subs({vars[0]: x_cur[0]}))
            x_new = x_cur - lr * grad_value
            stpIdx=i
            if abs(x_new - x_cur) < eps:
                break
            x_cur = x_new[0]
        elif ndim == 2:
            grad_value = np.array(grad.subs({vars[0]: x_cur[0], vars[1]:x_cur[1]}))
            x_new = x_cur - lr * grad_value.reshape((1,2))
            difference = x_new-x_cur
            norm=np.array([difference[0][0], difference[0][1]]).astype(float)
            stpIdx=i
            if np.linalg.norm(norm) < eps:
                break
            x_cur = [x_new[0][0],x_new[0][1]]

        x_hist.append(x_cur)

    return x_cur, np.array(x_hist), stpIdx
