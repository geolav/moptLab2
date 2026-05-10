import numpy as np
from typing import Callable
from scipy.optimize import line_search
from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_wolfe(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER
) -> OptResult:
    """Градиентный спуск с дроблением шага (сильные условия Вольфе).
    Используем scipy.optimize.line_search для поиска шага.
    """
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        d = -g

        alpha_res = line_search(
            lambda z: cf(z), lambda z: cg(z),
            x, d, gfk=g, c1=c1, c2=c2
        )
        alpha = alpha_res[0]

        if alpha is None or alpha < 1e-16:
            alpha = 1e-4

        x = x + alpha * d
        traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)
