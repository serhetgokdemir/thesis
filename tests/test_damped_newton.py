import pytest
from single_variable.damped_newton import damped_newton

# f(x) = x^3 - x - 2, kök ~1.5213797
f1 = lambda x: x**3 - x - 2
df1 = lambda x: 3*x**2 - 1

def test_damped_newton_fixed_alpha():
    # alpha küçük seçilirse daha yavaş ama kararlı yakınsar
    result = damped_newton(f1, df1, 0.5, alpha=0.5)
    assert result["method"] == "damped_newton"
    assert result["converged"]
    assert abs(result["root"] - 1.5213797) < 1e-7
    assert isinstance(result["iterations"], int)
    assert result["iterations"] > 0
    assert isinstance(result["final_residual"], float)
    assert isinstance(result["final_error"], float)
    assert len(result["history"]) == result["iterations"]
    assert result["history"][0]["alpha"] == 0.5

def test_damped_newton_backtracking():
    # kötü bir başlangıç noktasından backtracking ile kurtulmayı dener
    result = damped_newton(f1, df1, 0, backtracking=True, alpha=1.0)
    assert result["method"] == "damped_newton"
    assert result["converged"]
    assert abs(result["root"] - 1.5213797) < 1e-7
    assert isinstance(result["iterations"], int)
    assert result["iterations"] > 0
    assert isinstance(result["final_residual"], float)
    assert isinstance(result["final_error"], float)
    assert len(result["history"]) == result["iterations"]
    # ilk iterasyonda alpha'nın 1.0'dan küçük olmasını bekleriz
    assert result["history"][0]["alpha"] < 1.0

def test_damped_newton_no_convergence():
    # bu başlangıç noktası ve sabit alpha ile ıraksayabilir
    result = damped_newton(f1, df1, 0, max_iter=10, alpha=1.0, backtracking=False)
    assert result["method"] == "damped_newton"
    assert not result["converged"]
    assert result["message"] == "maximum iterations reached"
    assert result["iterations"] == 10
    assert isinstance(result["final_residual"], float)
    assert isinstance(result["final_error"], float)
    assert len(result["history"]) == result["iterations"]

def test_damped_newton_history_keys():
    result = damped_newton(f1, df1, 1.5, backtracking=True)
    history_item = result["history"][0]
    expected_keys = {"iteration", "x", "fx", "dfx", "step", "alpha", "error"}
    assert expected_keys == set(history_item.keys())
    assert isinstance(history_item["x"], (int, float))
    assert isinstance(history_item["fx"], (int, float))
    assert isinstance(history_item["dfx"], (int, float))
