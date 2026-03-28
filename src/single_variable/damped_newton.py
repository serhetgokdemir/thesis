from typing import Callable, Dict, List, Any

def damped_newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-8,
    max_iter: int = 100,
    alpha: float = 1.0,
    backtracking: bool = False,
    min_alpha: float = 1e-4,
    rho: float = 0.5
) -> Dict[str, Any]:
    """
    finds a root of a function using the damped newton-raphson method.

    args:
        f: the function to find the root of.
        df: the derivative of the function.
        x0: the initial guess.
        tol: the tolerance for convergence.
        max_iter: the maximum number of iterations.
        alpha: the initial damping factor (step size).
        backtracking: whether to use backtracking line search to find alpha.
        min_alpha: the minimum value for alpha in backtracking.
        rho: the factor by which to decrease alpha in backtracking.

    returns:
        a dictionary containing the results of the damped newton-raphson method.
    """
    x = x0
    history: List[Dict[str, Any]] = []
    current_alpha = alpha
    last_error: float | None = None

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-15:
            return {
                "method": "damped_newton",
                "root": x,
                "converged": False,
                "iterations": i,
                "final_error": None,
                "final_residual": abs(fx),
                "message": "derivative too small",
                "history": history
            }

        # her iterasyonda alpha'yi temiz bir sekilde baslat
        current_alpha = alpha
        step_direction = fx / dfx

        # backtracking line search
        if backtracking:
            while abs(f(x - current_alpha * step_direction)) >= abs(fx) and current_alpha > min_alpha:
                current_alpha *= rho
        
        step = current_alpha * step_direction
        x_next = x - step
        error = abs(step)
        last_error = error

        history.append({
            "iteration": i,
            "x": x,
            "fx": fx,
            "dfx": dfx,
            "step": step,
            "alpha": current_alpha,
            "error": error
        })

        f_next = f(x_next)
        residual_next = abs(f_next)

        if residual_next <= tol:
            return {
                "method": "damped_newton",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": error,
                "final_residual": residual_next,
                "message": "converged by residual tolerance",
                "history": history
            }

        if error <= tol:
            return {
                "method": "damped_newton",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": error,
                "final_residual": residual_next,
                "message": "converged by step tolerance",
                "history": history
            }
            
        x = x_next

    return {
        "method": "damped_newton",
        "root": x,
        "converged": False,
        "iterations": max_iter,
        "final_error": last_error,
        "final_residual": abs(f(x)),
        "message": "maximum iterations reached",
        "history": history
    }
