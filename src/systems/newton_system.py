import numpy as np
from typing import Callable, Dict, List, Any

def newton_system(
    F: Callable[[np.ndarray], np.ndarray],
    J: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    solves a system of nonlinear equations using newton's method.

    args:
        F: the system of equations.
        J: the jacobian of the system.
        x0: the initial guess.
        tol: the tolerance for convergence.
        max_iter: the maximum number of iterations.

    returns:
        a dictionary containing the results of newton's method for systems.
    """
    x = np.array(x0, dtype=float)
    history: List[Dict[str, Any]] = []

    for i in range(max_iter):
        Fx = F(x)
        norm_Fx = np.linalg.norm(Fx)

        if norm_Fx <= tol:
            return {
                "method": "newton_system",
                "root": x,
                "converged": True,
                "iterations": i,
                "final_error": 0,
                "final_residual": norm_Fx,
                "message": "converged by residual tolerance (initial guess)",
                "history": history
            }

        try:
            Jx = J(x)
            s = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            return {
                "method": "newton_system",
                "root": x,
                "converged": False,
                "iterations": i,
                "final_error": None,
                "final_residual": norm_Fx,
                "message": "jacobian solve failed",
                "history": history
            }

        step_norm = np.linalg.norm(s)
        x_next = x + s
        
        history.append({
            "iteration": i,
            "x": x.tolist(),
            "norm_f": norm_Fx,
            "step_norm": step_norm,
            "error": step_norm
        })

        if np.linalg.norm(F(x_next)) <= tol:
            return {
                "method": "newton_system",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": step_norm,
                "final_residual": np.linalg.norm(F(x_next)),
                "message": "converged by residual tolerance",
                "history": history
            }

        if step_norm <= tol:
            return {
                "method": "newton_system",
                "root": x_next,
                "converged": True,
                "iterations": i + 1,
                "final_error": step_norm,
                "final_residual": np.linalg.norm(F(x_next)),
                "message": "converged by step tolerance",
                "history": history
            }
            
        x = x_next

    return {
        "method": "newton_system",
        "root": x,
        "converged": False,
        "iterations": max_iter,
        "final_error": np.linalg.norm(s),
        "final_residual": np.linalg.norm(F(x)),
        "message": "maximum iterations reached",
        "history": history
    }
