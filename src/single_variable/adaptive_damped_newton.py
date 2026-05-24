from typing import Callable, Dict, List, Any


def adaptive_damped_newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-8,
    max_iter: int = 100,
    c: float = 1e-4,
    rho: float = 0.5,
    min_alpha: float = 1e-8
) -> Dict[str, Any]:
    """
    Finds a root of a nonlinear equation using Newton's method
    with Armijo backtracking line search.
    """

    def phi(x: float) -> float:
        return 0.5 * f(x) ** 2

    x = x0
    history: List[Dict[str, Any]] = []
    last_error = None

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-15:
            return {
                "method": "adaptive_damped_newton",
                "root": x,
                "converged": False,
                "iterations": i,
                "final_error": last_error,
                "final_residual": abs(fx),
                "message": "derivative too small",
                "history": history
            }

        step_direction = -fx / dfx
        alpha = 1.0

        phi_x = phi(x)
        phi_prime_x = fx * dfx

        while (
            phi(x + alpha * step_direction)
            > phi_x + c * alpha * phi_prime_x * step_direction
            and alpha > min_alpha
        ):
            alpha *= rho

        x_next = x + alpha * step_direction
        error = abs(x_next - x)
        residual_next = abs(f(x_next))
        last_error = error

        history.append({
            "iteration": i,
            "x": x,
            "fx": fx,
            "dfx": dfx,
            "alpha": alpha,
            "step": alpha * step_direction,
            "error": error,
            "residual": abs(fx),
            "phi": phi_x
        })

        if residual_next <= tol:
            return {
                "method": "adaptive_damped_newton",
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
                "method": "adaptive_damped_newton",
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
        "method": "adaptive_damped_newton",
        "root": x,
        "converged": False,
        "iterations": max_iter,
        "final_error": last_error,
        "final_residual": abs(f(x)),
        "message": "maximum iterations reached",
        "history": history
    }