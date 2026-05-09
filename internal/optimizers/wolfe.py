import numpy as np
from typing import Callable
from ..utils.result  import OptResult
from ..utils.counter import CallCounter

MAX_ITER = 100_000


def gradient_descent_wolfe(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:
    """Градиентный спуск с дроблением шага (сильные условия Вольфе)."""
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        p = -g
        alpha = alpha0
        fk = cf(x)

        for _ in range(60):
            x_try = x + alpha * p
            f_try = cf(x_try)
            g_try = cg(x_try)
            armijo_cond = f_try <= fk + c1 * alpha * np.dot(g, p)
            wolfe_cond = abs(np.dot(g_try, p)) <= c2 * abs(np.dot(g, p))
            if armijo_cond and wolfe_cond:
                break
            alpha *= 0.5

        # Fallback: Армихо если Вольфе не нашёл шаг
        if alpha is None or alpha < 1e-15 or not np.isfinite(alpha):
            fk = cf(x)
            alpha = alpha0
            gn = np.linalg.norm(g)
            for _ in range(60):
                if cf(x - alpha * g) <= fk - c1 * alpha * gn**2:
                    break
                alpha *= 0.5

        x = x + alpha * p
        traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)