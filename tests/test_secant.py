import pytest
from src.single_variable.secant import secant

# f(x) = x^3 - x - 2, kök ~1.5213797
f1 = lambda x: x**3 - x - 2
# f(x) = cos(x) - x, kök ~0.739085
f2 = lambda x: pytest.importorskip("numpy").cos(x) - x

def test_secant_convergence():
    result = secant(f1, 1, 2)
    assert result["converged"]
    assert "root" in result
    assert abs(result["root"] - 1.5213797) < 1e-5
    assert result["final_residual"] < 1e-8
    assert len(result["history"]) > 0

def test_secant_convergence_other_func():
    result = secant(f2, 0, 1)
    assert result["converged"]
    assert abs(result["root"] - 0.739085) < 1e-5

def test_secant_small_denominator():
    # bu fonksiyonun türevi x=0'da sıfır, secant zorlanabilir
    f_flat = lambda x: x**2
    result = secant(f_flat, -0.1, 0.1, max_iter=10) # x0 ve x1 simetrik
    # bu testin tam davranışı implementasyona bağlı, ama "small denominator" bekleriz
    # eğer f0==f1 olursa bu hata tetiklenir
    result_fail = secant(lambda x: 5, 0, 1)
    assert "denominator too small" in result_fail["message"]


def test_secant_max_iter():
    result = secant(f1, 1, 2, max_iter=3)
    assert not result["converged"]
    assert result["message"] == "maximum iterations reached"

def test_secant_history_keys():
    result = secant(f1, 1, 2)
    history_item = result["history"][0]
    expected_keys = {"iteration", "x_prev", "x_curr", "f_prev", "f_curr", "x_next", "error"}
    assert expected_keys == set(history_item.keys())
