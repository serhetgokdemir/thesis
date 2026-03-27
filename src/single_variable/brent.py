from typing import Callable, Dict, Any, List

from math import copysign


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Dict[str, Any]:
    """finds a root of a function using brent's method.

    this implementation follows the classical brent algorithm,
    combining bisection, secant and inverse quadratic interpolation.
    """

    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError("invalid bracketing interval: f(a) and f(b) must have opposite signs.")

    # ensure |f(b)| <= |f(a)|
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    history: List[Dict[str, Any]] = []

    eps = 2.220446049250313e-16  # double precision machine epsilon
    # use a slightly stricter internal tolerance to get a more accurate root
    tol_internal = tol * 0.1

    for i in range(max_iter):
        # record iteration state
        history.append(
            {
                "iteration": i,
                "a": a,
                "b": b,
                "c": c,
                "fa": fa,
                "fb": fb,
                "fc": fc,
                "interval_width": abs(c - b),
            }
        )

        # make sure b is the best approximation so far
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        # convergence checks: bracket width and residual
        m = 0.5 * (c - b)
        tol_act = 2.0 * eps * abs(b) + tol_internal

        if abs(m) <= tol_act and abs(fb) <= tol_internal:
            return {
                "method": "brent",
                "root": b,
                "converged": True,
                "iterations": i + 1,
                "final_error": abs(m),
                "final_residual": abs(fb),
                "message": "converged by tolerance",
                "history": history,
            }

        # decide whether to use interpolation or bisection
        if abs(e) >= tol_act and abs(fa) > abs(fb):
            # attempt inverse quadratic interpolation or secant
            s = fb / fa
            if a == c:
                # secant step
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # inverse quadratic interpolation
                q_ = fa / fc
                r_ = fb / fc
                p = s * (2.0 * m * q_ * (q_ - r_) - (b - a) * (r_ - 1.0))
                q = (q_ - 1.0) * (r_ - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            p = abs(p)

            # accept interpolation only if it is safe
            if 2.0 * p < min(3.0 * m * q - abs(tol_act * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = m
                e = m
        else:
            # fall back to bisection
            d = m
            e = m

        # update a and evaluate new point
        a = b
        fa = fb

        if abs(d) > tol_act:
            b = b + d
        else:
            b = b + copysign(tol_act, m)
        fb = f(b)

        # keep bracket [b, c] such that f(b) and f(c) have opposite signs
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c = a
            fc = fa
            d = b - a
            e = d

    return {
        "method": "brent",
        "root": b,
        "converged": False,
        "iterations": max_iter,
        "final_error": abs(c - b) / 2.0,
        "final_residual": abs(fb),
        "message": "maximum iterations reached",
        "history": history,
    }
