import numpy as np

class Himmelblau:
    """Функция Химмельблау: (x²+y-11)² + (x+y²-7)²  — 4 локальных минимума"""
    name = "Himmelblau"
    x_opt = np.array([3.0, 2.0])

    def f(self, x: np.ndarray) -> float:
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def grad(self, x: np.ndarray) -> np.ndarray:
        dfdx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
        dfdy = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
        return np.array([dfdx, dfdy])
