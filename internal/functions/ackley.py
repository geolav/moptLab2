import numpy as np


class Ackley:
    """Функция Экли (многоэкстремальная), x* = (0,0)"""
    name  = "Ackley"
    x_opt = np.zeros(2)

    def f(self, x: np.ndarray) -> float:
        r  = np.sqrt(0.5*(x[0]**2 + x[1]**2))
        s2 = 0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
        return -20*np.exp(-0.2*r) - np.exp(s2) + 20 + np.e

    def grad(self, x: np.ndarray) -> np.ndarray:
        r  = np.sqrt(0.5*(x[0]**2 + x[1]**2)) + 1e-12
        s2 = 0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
        e1 = np.exp(-0.2*r)
        e2 = np.exp(s2)
        return np.array([
            20*0.2*x[0]/(2*r)*e1 + np.pi*np.sin(2*np.pi*x[0])*e2,
            20*0.2*x[1]/(2*r)*e1 + np.pi*np.sin(2*np.pi*x[1])*e2,
        ])