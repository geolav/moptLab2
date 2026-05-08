import numpy as np
from typing import Callable
from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_armijo(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    beta: float = 0.5,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER
) -> OptResult:
    """Градиентный спуск с дроблением шага (условие Армихо)."""
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        fk = cf(x)
        alpha = alpha0

        while cf(x - alpha * g) > fk - c1 * alpha * np.dot(g, g):
            alpha *= beta
            if alpha < 1e-16:
                break

        x = x - alpha * g
        traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)
