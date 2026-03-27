from typing import Callable, Dict, List, Any

def newton(
    f: Callable[[float], float], 
    df: Callable[[float], float], 
    x0: float, 
    tol: float = 1e-8, 
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    finds a root of a function using the newton-raphson method.

    args:
        f: the function to find the root of.
        df: the derivative of the function.
        x0: the initial guess.
        tol: the tolerance for convergence.
        max_iter: the maximum number of iterations.

    returns:
        a dictionary containing the results of the newton-raphson method.
    """
    x = x0
    history: List[Dict[str, Any]] = []

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        # handle near-zero derivative to avoid unstable steps
        if abs(dfx) < max(10.0 * tol, 1e-15):
            return {
                "method": "newton",
                "root": x,
                "converged": False,
                "iterations": i,
                "final_error": None,
                "final_residual": abs(fx),
                "message": "derivative too small",
                "history": history
            }

        step = fx / dfx
        x_next = x - step
        error = abs(step)

        history.append({
            "iteration": i,
            "x": x,
            "fx": fx,
            "dfx": dfx,
            "step": step,
            "alpha": 1.0,
            "error": error
        })
        
        if error <= tol:
            return {
                "method": "newton",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": error,
                "final_residual": abs(f(x_next)),
                "message": "converged by step tolerance",
                "history": history
            }

        if abs(f(x_next)) <= tol:
            return {
                "method": "newton",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": error,
                "final_residual": abs(f(x_next)),
                "message": "converged by residual tolerance",
                "history": history
            }
        
        x = x_next

    return {
        "method": "newton",
        "root": x,
        "converged": False,
        "iterations": max_iter,
        "final_error": abs(step),
        "final_residual": abs(f(x)),
        "message": "maximum iterations reached",
        "history": history
    }
