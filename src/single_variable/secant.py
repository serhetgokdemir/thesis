from typing import Callable, Dict, List, Any

def secant(
    f: Callable[[float], float], 
    x0: float, 
    x1: float, 
    tol: float = 1e-8, 
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    finds a root of a function using the secant method.

    args:
        f: the function to find the root of.
        x0: the first initial guess.
        x1: the second initial guess.
        tol: the tolerance for convergence.
        max_iter: the maximum number of iterations.

    returns:
        a dictionary containing the results of the secant method.
    """
    f0 = f(x0)
    f1 = f(x1)
    history: List[Dict[str, Any]] = []

    for i in range(max_iter):
        denominator = f1 - f0
        if abs(denominator) < 1e-15:
            return {
                "method": "secant",
                "root": x1,
                "converged": False,
                "iterations": i,
                "final_error": None,
                "final_residual": abs(f1),
                "message": "secant denominator too small",
                "history": history
            }

        x_next = x1 - f1 * (x1 - x0) / denominator
        error = abs(x_next - x1)
        
        history.append({
            "iteration": i,
            "x_prev": x0,
            "x_curr": x1,
            "f_prev": f0,
            "f_curr": f1,
            "x_next": x_next,
            "error": error
        })

        f_next = f(x_next)

        if abs(f_next) <= tol:
            return {
                "method": "secant",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": error,
                "final_residual": abs(f_next),
                "message": "converged by residual tolerance",
                "history": history
            }

        if error <= tol:
            return {
                "method": "secant",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": error,
                "final_residual": abs(f_next),
                "message": "converged by step tolerance",
                "history": history
            }

        x0, x1 = x1, x_next
        f0, f1 = f1, f_next

    return {
        "method": "secant",
        "root": x1,
        "converged": False,
        "iterations": max_iter,
        "final_error": abs(x1 - x0),
        "final_residual": abs(f1),
        "message": "maximum iterations reached",
        "history": history
    }
