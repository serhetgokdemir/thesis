import pytest
from src.single_variable.brent import brent

# f(x) = x^3 - x - 2, kök ~1.5213797
f1 = lambda x: x**3 - x - 2
# f(x) = cos(x) - x, kök ~0.739085
f2 = lambda x: pytest.importorskip("numpy").cos(x) - x

def test_brent_convergence():
    result = brent(f1, 1, 2)
    assert result["converged"]
    assert "root" in result
    assert abs(result["root"] - 1.5213797) < 1e-7
    assert result["final_residual"] < 1e-8
    assert len(result["history"]) > 0

def test_brent_convergence_other_func():
    result = brent(f2, 0, 1)
    assert result["converged"]
    assert abs(result["root"] - 0.739085) < 1e-7

def test_brent_invalid_interval():
    with pytest.raises(ValueError, match="invalid bracketing interval"):
        brent(f1, 2, 3)

def test_brent_max_iter():
    result = brent(f1, 1, 2, max_iter=3)
    assert not result["converged"]
    assert result["message"] == "maximum iterations reached"

def test_brent_history_keys():
    result = brent(f1, 1, 2)
    history_item = result["history"][0]
    expected_keys = {"iteration", "a", "b", "c", "fa", "fb", "fc", "interval_width"}
    assert set(history_item.keys()).issubset(expected_keys)
