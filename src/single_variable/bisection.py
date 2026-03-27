from typing import Callable, Dict, List, Any

def bisection(
    f: Callable[[float], float], 
    a: float, 
    b: float, 
    tol: float = 1e-8, 
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    finds a root of a function using the bisection method.
    
    args:
        f: the function to find the root of.
        a: the left side of the interval.
        b: the right side of the interval.
        tol: the tolerance for convergence.
        max_iter: the maximum number of iterations.
        
    returns:
        a dictionary containing the results of the bisection method.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError("invalid bracketing interval: f(a) and f(b) must have opposite signs.")

    history: List[Dict[str, Any]] = []
    c = 0.0
    fc = 0.0

    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        interval_width = abs(b - a)

        history.append({
            "iteration": i,
            "a": a,
            "b": b,
            "c": c,
            "fa": fa,
            "fb": fb,
            "fc": fc,
            "interval_width": interval_width
        })

        if abs(fc) <= tol:
            return {
                "method": "bisection",
                "root": c,
                "converged": True,
                "iterations": i + 1,
                "final_error": interval_width / 2,
                "final_residual": abs(fc),
                "message": "converged by residual tolerance",
                "history": history
            }

        if interval_width <= tol:
            return {
                "method": "bisection",
                "root": c,
                "converged": True,
                "iterations": i + 1,
                "final_error": interval_width / 2,
                "final_residual": abs(fc),
                "message": "converged by interval width",
                "history": history
            }

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return {
        "method": "bisection",
        "root": c,
        "converged": False,
        "iterations": max_iter,
        "final_error": abs(b - a) / 2,
        "final_residual": abs(fc),
        "message": "maximum iterations reached",
        "history": history
    }
