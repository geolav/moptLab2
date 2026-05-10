import numpy as np

class Rosenbrock:
    """Функция Розенброка: f(x,y) = (1-x)² + 100(y-x²)² — глобальный минимум в (1,1)"""
    name = "Rosenbrock"
    x_opt = np.array([1.0, 1.0])

    def f(self, x: np.ndarray) -> float:
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad(self, x: np.ndarray) -> np.ndarray:
        dfdx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dfdy = 200*(x[1] - x[0]**2)
        return np.array([dfdx, dfdy])
