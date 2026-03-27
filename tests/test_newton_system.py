import pytest
import numpy as np
from src.systems.newton_system import newton_system

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
    
    assert result["converged"]
    root = result["root"]
    assert isinstance(root, np.ndarray)
    assert np.allclose(root, [1.82287565, 0.82287565], atol=1e-5)
    assert result["final_residual"] < 1e-8
    assert len(result["history"]) > 0

def test_newton_system_singular_jacobian():
    # bu sistemin jacobian'ı [0,0]'da singular
    F_singular = lambda x: np.array([x[0]**2, x[1]])
    J_singular = lambda x: np.array([[2*x[0], 0], [0, 1]])
    x0 = np.array([0.0, 2.0])
    
    result = newton_system(F_singular, J_singular, x0)
    assert not result["converged"]
    assert result["message"] == "jacobian solve failed"

def test_newton_system_max_iter():
    x0 = np.array([-10.0, 10.0]) # uzak bir başlangıç
    result = newton_system(F1, J1, x0, max_iter=5)
    assert not result["converged"]
    assert result["message"] == "maximum iterations reached"

def test_newton_system_history_keys():
    x0 = np.array([2.0, 1.0])
    result = newton_system(F1, J1, x0)
    history_item = result["history"][0]
    expected_keys = {"iteration", "x", "norm_f", "step_norm", "error"}
    assert expected_keys == set(history_item.keys())
