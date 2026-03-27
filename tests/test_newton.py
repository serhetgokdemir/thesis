import pytest
from single_variable.newton import newton

# f(x) = x^3 - x - 2, kök ~1.5213797
f1 = lambda x: x**3 - x - 2
df1 = lambda x: 3*x**2 - 1

def test_newton_convergence():
    result = newton(f1, df1, 1.5)
    assert result["converged"]
    assert "root" in result
    assert abs(result["root"] - 1.5213797) < 1e-7
    assert result["final_residual"] < 1e-8
    assert len(result["history"]) > 0

def test_newton_small_derivative():
    # f(x) = x^2, kök 0, türev 0'da 0
    f_quad = lambda x: x**2
    df_quad = lambda x: 2*x
    # başlangıç noktası 0'a çok yakınsa türev küçük olur
    result = newton(f_quad, df_quad, 1e-9)
    assert not result["converged"]
    assert result["message"] == "derivative too small"

def test_newton_max_iter():
    # bu başlangıç noktası ıraksar
    result = newton(f1, df1, 0, max_iter=5)
    assert not result["converged"]
    assert result["message"] == "maximum iterations reached"

def test_newton_history_keys():
    result = newton(f1, df1, 1.5)
    history_item = result["history"][0]
    expected_keys = {"iteration", "x", "fx", "dfx", "step", "alpha", "error"}
    assert expected_keys == set(history_item.keys())
