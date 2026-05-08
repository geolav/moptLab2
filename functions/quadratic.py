import numpy as np

class QuadraticWellConditioned:
    """Хорошо обусловленная квадратичная функция: f(x,y) = x² + y²  (κ=1)"""
    name = "Quadratic (well-cond, κ=1)"
    A = np.array([[2.0, 0.0], [0.0, 2.0]])
    x_opt = np.array([0.0, 0.0])

    def f(self, x: np.ndarray) -> float:
        return float(x @ self.A @ x / 2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def hessian(self) -> np.ndarray:
        return self.A


class QuadraticIllConditioned:
    """Плохо обусловленная квадратичная функция: f(x,y) = 50x² + 0.5y²  (κ=100)"""
    name = "Quadratic (ill-cond, κ=100)"
    A = np.array([[100.0, 0.0], [0.0, 1.0]])
    x_opt = np.array([0.0, 0.0])

    def f(self, x: np.ndarray) -> float:
        return float(x @ self.A @ x / 2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def hessian(self) -> np.ndarray:
        return self.A
