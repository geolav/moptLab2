import numpy as np
from dataclasses import dataclass, field


@dataclass
class OptResult:
    x:          np.ndarray
    f_val:      float
    n_iter:     int
    n_f:        int
    n_grad:     int
    converged:  bool
    trajectory: list = field(default_factory=list)

    def __str__(self):
        status = "✓" if self.converged else "✗"
        return (f"{status} iter={self.n_iter:5d}  f_calls={self.n_f:6d}  "
                f"grad_calls={self.n_grad:5d}  f(x*)={self.f_val:.4e}  "
                f"x*={np.round(self.x, 6)}")