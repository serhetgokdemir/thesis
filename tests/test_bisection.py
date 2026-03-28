import pytest
from single_variable.bisection import bisection

# f(x) = x^3 - x - 2, kök ~1.5213797
f1 = lambda x: x**3 - x - 2

def test_bisection_convergence():
    result = bisection(f1, 1, 2)
    # temel şema ve yakınsama kontrolleri
    assert result["method"] == "bisection"
    assert result["converged"]
    assert "root" in result
    assert isinstance(result["iterations"], int)
    assert result["iterations"] > 0
    assert abs(result["root"] - 1.5213797) < 1e-5
    assert isinstance(result["final_residual"], float)
    assert isinstance(result["final_error"], float)
    assert result["final_residual"] < 1e-8
    assert len(result["history"]) == result["iterations"]

def test_bisection_invalid_interval():
    with pytest.raises(ValueError, match="invalid bracketing interval"):
        bisection(f1, 2, 3)

def test_bisection_max_iter():
    # deterministik maksimum iterasyon senaryosu: limit küçük
    result = bisection(f1, 1, 2, max_iter=1)
    assert result["method"] == "bisection"
    assert not result["converged"]
    assert result["message"] == "maximum iterations reached"
    assert result["iterations"] == 1
    assert isinstance(result["final_error"], float)
    assert isinstance(result["final_residual"], float)
    assert len(result["history"]) == result["iterations"]

def test_bisection_history_keys():
    result = bisection(f1, 1, 2)
    history_item = result["history"][0]
    expected_keys = {"iteration", "a", "b", "c", "fa", "fb", "fc", "interval_width"}
    assert expected_keys == set(history_item.keys())
    # skaler tip kontrolleri
    assert isinstance(history_item["a"], (int, float))
    assert isinstance(history_item["b"], (int, float))
    assert isinstance(history_item["c"], (int, float))
    assert isinstance(history_item["fa"], (int, float))
    assert isinstance(history_item["fb"], (int, float))
    assert isinstance(history_item["fc"], (int, float))
