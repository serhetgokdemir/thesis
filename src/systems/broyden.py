import numpy as np
from typing import Callable, Dict, List, Any, Optional


def broyden(
    F: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    B0: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Dict[str, Any]:
    """
    solves a system of nonlinear equations using broyden's method.

    args:
        F: the system of equations.
        x0: the initial guess.
        B0: the initial jacobian approximation. if none, identity is used.
        tol: the tolerance for convergence.
        max_iter: the maximum number of iterations.

    returns:
        a dictionary containing the results of broyden's method.
    """
    x = np.array(x0, dtype=float)

    if B0 is None:
        B = np.eye(len(x))
    else:
        B = np.array(B0, dtype=float)
        if B.shape != (x.size, x.size):
            raise ValueError("initial jacobian approximation has incompatible shape")

    history: List[Dict[str, Any]] = []
    Fx = F(x)
    last_step_norm: float | None = None

    for i in range(max_iter):
        norm_Fx = float(np.linalg.norm(Fx))

        if norm_Fx <= tol:
            return {
                "method": "broyden",
                "root": x,
                "converged": True,
                "iterations": i,
                "final_error": 0.0,
                "final_residual": norm_Fx,
                "message": "converged by residual tolerance",
                "history": history,
            }

        try:
            s = np.linalg.solve(B, -Fx)
        except np.linalg.LinAlgError:
            return {
                "method": "broyden",
                "root": x,
                "converged": False,
                "iterations": i,
                "final_error": last_step_norm,
                "final_residual": norm_Fx,
                "message": "jacobian approximation solve failed",
                "history": history,
            }

        step_norm = float(np.linalg.norm(s))
        last_step_norm = step_norm
        x_next = x + s
        Fx_next = F(x_next)
        norm_Fx_next = float(np.linalg.norm(Fx_next))

        history.append(
            {
                "iteration": i,
                "x": x.tolist(),
                "norm_f": norm_Fx,
                "step_norm": step_norm,
                "error": step_norm,
            }
        )

        if norm_Fx_next <= tol:
            return {
                "method": "broyden",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": step_norm,
                "final_residual": norm_Fx_next,
                "message": "converged by residual tolerance",
                "history": history,
            }

        if step_norm <= tol:
            return {
                "method": "broyden",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": step_norm,
                "final_residual": norm_Fx_next,
                "message": "converged by step tolerance",
                "history": history,
            }

        # good broyden rank-1 update with safeguard on denominator
        y = Fx_next - Fx
        denom = float(s @ s)
        if denom <= 1e-16:
            return {
                "method": "broyden",
                "root": x_next,
                "converged": False,
                "iterations": i + 1,
                "final_error": step_norm,
                "final_residual": norm_Fx_next,
                "message": "broyden update denominator too small",
                "history": history,
            }

        B += np.outer((y - B @ s), s) / denom

        x = x_next
        Fx = Fx_next

    norm_Fx = float(np.linalg.norm(Fx))

    return {
        "method": "broyden",
        "root": x,
        "converged": False,
        "iterations": max_iter,
        "final_error": last_step_norm,
        "final_residual": norm_Fx,
        "message": "maximum iterations reached",
        "history": history,
    }
