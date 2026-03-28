import pytest
import numpy as np
from systems.newton_system import newton_system

# F(x, y) = [x^2 + y^2 - 4, x - y - 1]
# Köklerden biri ~[1.82287, 0.82287]
def F1(x):
    return np.array([
        x[0]**2 + x[1]**2 - 4,
        x[0] - x[1] - 1
    ])

def J1(x):
    return np.array([
        [2*x[0], 2*x[1]],
        [1, -1]
    ])

def test_newton_system_convergence():
    x0 = np.array([2.0, 1.0])
    result = newton_system(F1, J1, x0)
    
    # temel şema kontrolleri
    assert result["method"] == "newton_system"
    assert result["converged"]
    assert isinstance(result["iterations"], int)
    assert result["iterations"] > 0
    root = result["root"]
    assert isinstance(root, np.ndarray)
    assert np.allclose(root, [1.82287565, 0.82287565], atol=1e-5)
    # final hata ve residual tür/sınır kontrolleri
    assert isinstance(result["final_residual"], (float, np.floating))
    assert isinstance(result["final_error"], (float, np.floating))
    assert result["final_residual"] < 1e-8
    # history ile iterations tutarlılığı
    assert len(result["history"]) == result["iterations"]

def test_newton_system_singular_jacobian():
    # bu sistemin jacobian'ı [0,0]'da singular
    F_singular = lambda x: np.array([x[0]**2, x[1]])
    J_singular = lambda x: np.array([[2*x[0], 0], [0, 1]])
    x0 = np.array([0.0, 2.0])
    
    result = newton_system(F_singular, J_singular, x0)
    # temel şema ve failure türü
    assert result["method"] == "newton_system"
    assert not result["converged"]
    assert result["message"] == "jacobian solve failed"
    # iterasyon sayısı ve history tutarlılığı
    assert isinstance(result["iterations"], int)
    assert result["iterations"] == 0
    assert len(result["history"]) == result["iterations"]
    # hata ve residual tipleri
    assert result["final_error"] is None
    assert isinstance(result["final_residual"], (float, np.floating))

def test_newton_system_max_iter():
    # deterministik bir maksimum iterasyon senaryosu: çözüm yakın ama iterasyon limiti çok küçük
    x0 = np.array([2.0, 1.0])
    result = newton_system(F1, J1, x0, max_iter=1)

    assert result["method"] == "newton_system"
    assert not result["converged"]
    assert result["message"] == "maximum iterations reached"
    assert result["iterations"] == 1
    assert isinstance(result["final_error"], (float, np.floating))
    assert isinstance(result["final_residual"], (float, np.floating))
    # history ile iterations bire bir olmalı
    assert len(result["history"]) == result["iterations"]
    assert isinstance(result["root"], np.ndarray)

def test_newton_system_history_keys():
    x0 = np.array([2.0, 1.0])
    result = newton_system(F1, J1, x0)
    history_item = result["history"][0]
    expected_keys = {"iteration", "x", "norm_f", "step_norm", "error"}
    assert expected_keys == set(history_item.keys())
    # x içeriği list tipinde olmalı (tolist sonucu)
    assert isinstance(history_item["x"], list)
