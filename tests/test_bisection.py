import pytest
from src.single_variable.bisection import bisection

# f(x) = x^3 - x - 2, kök ~1.5213797
f1 = lambda x: x**3 - x - 2

def test_bisection_convergence():
    result = bisection(f1, 1, 2)
    assert result["converged"]
    assert "root" in result
    assert abs(result["root"] - 1.5213797) < 1e-5
    assert result["final_residual"] < 1e-8
    assert len(result["history"]) > 0

def test_bisection_invalid_interval():
    with pytest.raises(ValueError, match="invalid bracketing interval"):
        bisection(f1, 2, 3)

def test_bisection_max_iter():
    result = bisection(f1, 1, 2, max_iter=5)
    assert not result["converged"]
    assert result["message"] == "maximum iterations reached"

def test_bisection_history_keys():
    result = bisection(f1, 1, 2)
    history_item = result["history"][0]
    expected_keys = {"iteration", "a", "b", "c", "fa", "fb", "fc", "interval_width"}
    assert expected_keys == set(history_item.keys())
