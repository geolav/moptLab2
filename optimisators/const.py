from core.result import OptResult
from core.counter import CallCounter
from core.constants import MAX_ITER

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
        cf(x)  # один вызов f для контроля значения

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)