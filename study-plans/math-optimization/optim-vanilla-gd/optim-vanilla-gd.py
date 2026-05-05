import numpy as np

def vanilla_gradient_descent(x0, y0, lr, n_iters):
    """
    Returns: dict with 'trajectory' (list of [x,y] pairs), 'final_point' ([x,y]), 'final_value' (float)
    """

    
    x0_new=x0
    y0_new=y0
    def func(x0,y0):
        return x0**2+3*y0**2 
    lst=[[x0_new,y0_new]]
    for step in range(n_iters):
        dx=2*x0_new
        dy=6*y0_new
        
        x0_new=x0_new-lr*dx
        y0_new=y0_new-lr*dy
        lst.append([x0_new,y0_new])
        
        
    return {
            "trajectory": lst,
            "final_point": [x0_new,y0_new],
            "final_value": func(x0_new,y0_new)
            }
    
