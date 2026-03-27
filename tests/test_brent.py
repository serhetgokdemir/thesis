import math
import pytest
from single_variable.brent import brent

# f(x) = x^3 - x - 2, kök ~1.5213797
f1 = lambda x: x**3 - x - 2

# basit test fonksiyonu: f(x) = x^2 - 2, kök ~sqrt(2)
f_quad = lambda x: x**2 - 2.0
true_root_quad = math.sqrt(2.0)

def test_brent_convergence():
    result = brent(f1, 1, 2)
    assert result["converged"]
    assert "root" in result
    assert abs(result["root"] - 1.5213797) < 1e-7
    assert result["final_residual"] < 1e-8
    assert len(result["history"]) > 0

def test_brent_convergence_other_func():
    # basit fonksiyon f(x) = x^2 - 2 için ortak doğrulama testi
    result = brent(f_quad, 1, 2)
    assert result["converged"]
    assert abs(result["root"] - true_root_quad) < 1e-7

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
    expected_keys = {"iteration", "a", "b", "c", "x", "fa", "fb", "fc", "interval_width"}
    assert expected_keys == set(history_item.keys())
