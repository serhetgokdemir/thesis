from typing import Callable, Dict, Any

def brent(
    f: Callable[[float], float], 
    a: float, 
    b: float, 
    tol: float = 1e-8, 
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    finds a root of a function using brent's method.
    
    this is a careful implementation of brent's method that combines
    bisection, secant, and inverse quadratic interpolation.
    
    args:
        f: the function to find the root of.
        a: the left side of the interval.
        b: the right side of the interval.
        tol: the tolerance for convergence.
        max_iter: the maximum number of iterations.
        
    returns:
        a dictionary containing the results of brent's method.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError("invalid bracketing interval: f(a) and f(b) must have opposite signs.")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d
    
    history = []

    for i in range(max_iter):
        history.append({
            "iteration": i,
            "a": a, "b": b, "c": c,
            "fa": f(a), "fb": f(b), "fc": f(c),
            "interval_width": abs(b-a)
        })

        if abs(fb) <= tol:
            return {
                "method": "brent", "root": b, "converged": True, "iterations": i + 1,
                "final_error": abs(b-a), "final_residual": abs(fb),
                "message": "converged by residual tolerance", "history": history
            }
        
        if abs(b-a) <= tol:
             return {
                "method": "brent", "root": b, "converged": True, "iterations": i + 1,
                "final_error": abs(b-a), "final_residual": abs(fb),
                "message": "converged by interval width", "history": history
            }

        if fc is not None and fa is not None and fb != fc and fa != fc:
            # inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # secant method
            s = b - fb * (b - a) / (fb - fa)

        # check if interpolation is within bounds and efficient
        cond1 = (s < (3 * a + b) / 4 or s > b)
        cond2 = abs(s - b) >= abs(e) / 2
        cond3 = abs(e) < tol

        if cond1 or cond2 or cond3:
            # fallback to bisection
            s = (a + b) / 2
            d = s - b
            e = d
        else:
            d = s - b
            e = d

        # update points
        a, fa = b, fb
        b, fb = s, f(s)

        if fa * fb < 0:
            c, fc = a, fa
        else:
            c, fc = a, fa
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa


    return {
        "method": "brent", "root": b, "converged": False, "iterations": max_iter,
        "final_error": abs(b-a), "final_residual": abs(fb),
        "message": "maximum iterations reached", "history": history
    }
