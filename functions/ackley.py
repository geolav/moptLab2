import numpy as np

class Ackley:
    """Функция Экли (многоэкстремальная)"""
    name = "Ackley"
    x_opt = np.array([0.0, 0.0])

    def f(self, x: np.ndarray) -> float:
        a, b, c = 20, 0.2, 2*np.pi
        s1 = np.sqrt(0.5*(x[0]**2 + x[1]**2))
        s2 = 0.5*(np.cos(c*x[0]) + np.cos(c*x[1]))
        return -a*np.exp(-b*s1) - np.exp(s2) + a + np.e

    def grad(self, x: np.ndarray) -> np.ndarray:
        a, b, c = 20, 0.2, 2*np.pi
        r = np.sqrt(0.5*(x[0]**2 + x[1]**2)) + 1e-12
        s2 = 0.5*(np.cos(c*x[0]) + np.cos(c*x[1]))
        df = np.zeros(2)
        for i in range(2):
            df[i] = (a*b*x[i]/(2*r))*np.exp(-b*r) + np.pi*np.sin(c*x[i])*np.exp(s2)
        return df
