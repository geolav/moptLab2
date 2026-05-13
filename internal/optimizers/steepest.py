import numpy as np
from typing import Callable, Optional
from scipy.optimize import minimize_scalar
from ..utils.result  import OptResult
from ..utils.counter import CallCounter

MAX_ITER = 100_000


def gradient_descent_steepest(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    A:  Optional[np.ndarray] = None,
    eps:      float = 1e-8,
    max_iter: int   = MAX_ITER,
) -> OptResult:
    """Метод наискорейшего градиентного спуска.

    Если A задана (квадратичная f = ½xᵀAx) — аналитическая формула:
        α* = ‖g‖² / (gᵀAg)
    Иначе — точная одномерная минимизация через scipy.
    """
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x    = x0.copy().astype(float)
    traj = [x.copy()]

    for k in range(max_iter):
        g  = cg(x)
        gn = np.linalg.norm(g)
        if gn < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        if A is not None:
            denom = g @ A @ g
            alpha = gn**2 / denom if denom > 1e-16 else 1e-6
        else:
            res   = minimize_scalar(
                lambda a: cf(x - a * g),
                bounds=(1e-12, 50.0), method='bounded',
            )
            alpha = res.x

        x = x - alpha * g
        traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)