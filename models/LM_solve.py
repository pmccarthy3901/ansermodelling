import numpy as np 


def jacobian(f : callable,
             x : np.ndarray,
             h : float = 1e-5
        ) -> np.ndarray:
    '''
    Computes the Jacobian of a function at a point. 

    Parameters
    ---------- 
    f : callable 
        Function for which the jacobian is to be calculated
    x : np.ndarray 
        Point at which the jacobian is evaluated 
    h : float 
        Step size (default = 1e-5) 

    Returns 
    ------- 
    J : np.ndarray 
        Jacobian of f evaluated at x
    '''
    y0 = f(x)

    m = x.shape[0]
    n = y0.shape[0]

    J = np.zeros((n,m))

    for i in range(m):
        dx = np.zeros(m)
        dx[i] = h 
        J[:,i] = (f(x + dx) - y0) / h 

    return J


def lm_solve(f : callable,
             y : np.ndarray, 
             x0 : np.ndarray,
             lambda_0 : float = 1e-5,
             max_iter : int = 100,
             lambda_max : float = 1e7,
             eps_grad : float = 1e-5
        ) -> tuple[np.ndarray,list[np.ndarray]]:
    '''
    Implements the Levenberg-Marquardt algorithm for inverse solving. approximates x such that f(x) = y

    Parameters
    ---------- 
    f : callable 
        forward function 
    y : np.ndarray 
        y for which y = f(x)
    x0 : np.ndarray
        Initial guess for x
    lambda_0 : float 
        Initial damping coefficient 
    max_iter : int
        Maximum number of iterations 
    lambda_max : float 
        Maximum lambda before solver gives up
    eps_grad : float 
        Convergence stopping parameter
    Returns 
    -------
    x : np.ndarray 
        Output of Levenberg-Marquardt.
    xs : list[np.ndarray] 
        Sequence of x guesses
    '''

    x = x0.copy()
    r = f(x) - y
    lam = lambda_0 
    cost = r @ r 

    xs = []
    xs.append(x)


    for _ in range(max_iter):
        J = jacobian(f,x)
        JTJ = J.T @ J 
        JTr = J.T @ r 
        D = np.diag(np.maximum(np.diag(JTJ),1e-12))

        while True:
            delta = np.linalg.solve(JTJ + lam * D, -JTr)

            x_trial = x + delta
            r_trial = f(x_trial) - y 
            cost_trial = r_trial @ r_trial 

            #if guess is better make it next guess and weight closeness less
            if cost_trial < cost:
                x, r, cost = x_trial, r_trial, cost_trial
                xs.append(x)
                lam *= 0.1
                break 
            
            #weight closeness higher if it's failing
            else: 
                lam *= 10
                if lam > lambda_max:
                    return x, xs
        
        #break when convergence slows down sufficiently
        if np.linalg.norm(JTr, np.inf) < eps_grad: break 
        
    return x, xs
