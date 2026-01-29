import ufl

def circle(x):
    return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) - 0.25 * 0.25 + 1e-16

def star(x):
    c = [0.5, 0.5]
    n = 8.
    eps = 1e-9
    return 0.5 * (ufl.sqrt((x[0] - c[0])**2 + (x[1] - c[1])**2) - 0.1 * ufl.cos(n * ufl.atan((x[1] - c[1]) / ((x[0] - c[0]) + ufl.sign(x[0] - c[0] + eps) * eps)) + 2.) - 0.25)

def smooth_star(x):
	c = [0, 0]
	eps = 0.000000001
	n = 8
	return ((x[0]-c[0])**2+(x[1]-c[1])**2-(1+0.2*ufl.sin(n*ufl.atan((x[1]-c[1])/((x[0]-c[0])+ ufl.sign(x[0]-c[0]+eps)*eps)))))

def step(x):
    r = 0.25
    a_1 = 0.6
    a_2 = 0.3
    a_3 = 0
    res = ufl.conditional(ufl.le(x[0],0.5),-1,1)

    #res = ufl.conditional(ufl.le(x[0],1/3),a_1*x[0]-0.25,a_2* x[0]-0.15)
    #res = ufl.conditional(ufl.ge(x[0],2/3),a_3*x[0]+0.05,res)
	#res = ufl.max_value(-0.01*x[0]-0.005, -0.01*x[0]+0.005)
    return res 
    
def tan_function(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        return np.arctan((x[0] - 0.5) * 10)
    return ufl.atan((x[0] - 0.5) * 10)

def apply_noise(phi, mesh):
    import numpy as np
    from dolfinx import fem
    
    V_ls = phi.function_space
    
    # Generate noise
    noise = np.random.normal(0, 0.1, len(phi.x.array[:]))
    
    # Compute h_mesh (CellDiameter)
    h_mesh = ufl.CellDiameter(mesh)
    
    # Define xsi function to localize noise
    # The user's snippet:
    # xsi = ufl.conditional(ufl.lt(phi,-2*h_mesh+0.5),-1,0)
    # xsi = ufl.conditional(ufl.gt(phi,2*h_mesh+0.5),1,xsi)
    # Note: The user's snippet uses 'phi' in the conditional. 
    # Since phi is a Function, we can use it directly in UFL expressions.
    
    # However, we need to be careful about mixing Function and Expression for interpolation.
    # We need to project/interpolate xsi onto V_ls.
    
    xsi_expr_ufl = ufl.conditional(ufl.lt(phi, -2 * h_mesh + 0.5), -1, 0)
    xsi_expr_ufl = ufl.conditional(ufl.gt(phi, 2 * h_mesh + 0.5), 1, xsi_expr_ufl)
    
    xsi_expr = fem.Expression(xsi_expr_ufl, V_ls.element.interpolation_points())
    xsi = fem.Function(V_ls)
    xsi.interpolate(xsi_expr)
    
    # Apply noise
    # The user's snippet: phi.x.array[:] = phi.x.array+noise**2*xsi.x.array
    # Wait, noise**2 is always positive. And xsi is -1, 0, or 1.
    # So this adds positive noise where phi > ..., and subtracts where phi < ...
    # And 0 in between.
    phi.x.array[:] = phi.x.array + noise**2 * xsi.x.array
    phi.x.scatter_forward()
    
    return phi 
