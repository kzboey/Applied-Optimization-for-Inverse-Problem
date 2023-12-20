import numpy as np
import aomip

# Accept arbitrary operators to stack
class StackedOperator:
        
    def apply(self,opts,xs):
        block = []
        if len(opts) != len(xs):
            print("Must be same number of operator and arguments")
            return 0
        for opt,x in zip(opts, xs):
            block.append(opt.apply(x))
        return block
                     
    def applyAdjoint(self,opts,ys):
        block = []
        if len(opts) != len(ys):
            print("Must be same number of operator and arguments")
            return 0
        for opt,y in zip(opts, ys):
            block.append(opt.applyAdjoint(y))
        return block
    
    def addition(self,stack1,stack2):
        block = []
        if len(stack1) != len(stack2):
            print("Operator and x must be the same size to perform addition")
            return 0
        for s1,s2 in zip(stack1, stack2):
            block.append(s1+s2)
        return block
    
    def subtraction(self,stack1,stack2):
        block = []
        if len(stack1) != len(stack2):
            print("Operator and x must be the same size to perform addition")
            return 0
        for s1,s2 in zip(stack1, stack2):
            block.append(s1-s2)
        return block
    
    def scalar_mut(self,stack, scalar):
        block = []
        for s1 in stack:
            block.append(scalar * s1)
        return block

# Hard coded
# class StackedOperator:
        
#     def apply(self,A,gradOpt,x):
#         block = []
#         rawGrad = gradOpt.apply(x)
#         g1 = block.append(A.apply(x)) 
#         g2 = np.reshape(rawGrad, (rawGrad.shape[0], rawGrad.shape[1]*rawGrad.shape[2]))
#         return block
            
#     def applyAdjoint(self,A,gradOpt,y):
#         block = []
#         grad = gradOpt.applyAdjoint(y)
#         g1 = block.append(A.applyAdjoint(y)) 
#         g2 = block.append(grad[0,:,:])
#         g3 = block.append(grad[1,:,:])
#         return block
        
        
        
        