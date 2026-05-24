import pytest
from single_variable.adaptive_damped_newton import adaptive_damped_newton

# f(x) = x^3 - x - 2, root ~ 1.5213797
f1 = lambda x: x**3 - x - 2
df1 = lambda x: 3*x**2 - 1


def test_adaptive_damped_newton_converges():
    result = adaptive_damped_newton(f1, df1, 0.5)

    assert result["method"] == "adaptive_damped_newton"
    assert result["converged"]
    assert abs(result["root"] - 1.5213797) < 1e-7
    assert isinstance(result["iterations"], int)
    assert result["iterations"] > 0
    assert isinstance(result["final_residual"], float)
    assert isinstance(result["final_error"], float)
    assert len(result["history"]) == result["iterations"]


def test_adaptive_damped_newton_uses_armijo_backtracking():
    # Starting from x0 = 0 causes the full Newton step to oscillate.
    # Armijo backtracking should reduce alpha at least once.
    result = adaptive_damped_newton(f1, df1, 0.0)

    assert result["method"] == "adaptive_damped_newton"
    assert result["converged"]
    assert abs(result["root"] - 1.5213797) < 1e-7

    alphas = [item["alpha"] for item in result["history"]]
    assert any(alpha < 1.0 for alpha in alphas)


def test_adaptive_damped_newton_history_keys():
    result = adaptive_damped_newton(f1, df1, 1.5)

    history_item = result["history"][0]

    expected_keys = {
        "iteration",
        "x",
        "fx",
        "dfx",
        "alpha",
        "step",
        "error",
        "residual",
        "phi"
    }

    assert expected_keys == set(history_item.keys())
    assert isinstance(history_item["x"], (int, float))
    assert isinstance(history_item["fx"], (int, float))
    assert isinstance(history_item["dfx"], (int, float))
    assert isinstance(history_item["alpha"], float)
    assert isinstance(history_item["phi"], float)


def test_adaptive_damped_newton_derivative_too_small():
    f = lambda x: x**3
    df = lambda x: 3*x**2

    result = adaptive_damped_newton(f, df, 0.0)

    assert result["method"] == "adaptive_damped_newton"
    assert not result["converged"]
    assert result["message"] == "derivative too small"
    assert result["iterations"] == 0
    assert isinstance(result["final_residual"], float)