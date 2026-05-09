import numpy as np
from typing import Callable
from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_const(
    f: Callable, grad: Callable,
    x0: np.ndarray, alpha: float,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER
) -> OptResult:
    """Градиентный спуск с постоянным шагом."""
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)
        x = x - alpha * g
        traj.append(x.copy())
        cf(x)

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)


# import numpy as np
# from typing import Callable
# from ..utils.result  import OptResult
# from ..utils.counter import CallCounter
#
# MAX_ITER = 100_000
#
#
# def gradient_descent_const(
#     f: Callable, grad: Callable,
#     x0: np.ndarray, alpha: float,
#     eps: float = 1e-8,
#     max_iter: int = MAX_ITER,
# ) -> OptResult:
#     """Градиентный спуск с постоянным шагом."""
#     cf = CallCounter(f)
#     cg = CallCounter(grad)
#
#     x    = x0.copy().astype(float)
#     traj = [x.copy()]
#
#     for k in range(max_iter):
#         g = cg(x)
#         if np.linalg.norm(g) < eps:
#             return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)
#
#         x_new = x - alpha * g
#         # Проверка на расходимость — прекращаем без лишних предупреждений
#         if not np.all(np.isfinite(x_new)):
#             return OptResult(x_new, float('nan'), k, cf.count, cg.count, False, traj)
#
#         x = x_new
#         traj.append(x.copy())
#
#     return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)