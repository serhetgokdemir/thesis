from typing import Callable, Dict, List, Any


def damped_newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-8,
    max_iter: int = 100,
    alpha: float = 0.8
) -> Dict[str, Any]:
    """
    finds a root of a function using the damped newton-raphson method.
    """

    x = x0
    history: List[Dict[str, Any]] = []
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

        step = alpha * fx / dfx
        x_next = x - step

        error = abs(step)
        last_error = error

        history.append({
            "iteration": i,
            "x": x,
            "fx": fx,
            "dfx": dfx,
            "step": step,
            "alpha": alpha,
            "error": error
        })

        residual_next = abs(f(x_next))

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