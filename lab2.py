"""
Лабораторная работа 2: Градиентный спуск — классика
Методы оптимизации (КТ'26)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize_scalar
from dataclasses import dataclass, field
from typing import Callable, Optional
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  Счётчик вызовов (декоратор)
# ─────────────────────────────────────────────────────────────

class CallCounter:
    """Оборачивает функцию и считает число вызовов."""
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

    def reset(self):
        self.count = 0


# ─────────────────────────────────────────────────────────────
#  Результат оптимизации
# ─────────────────────────────────────────────────────────────

@dataclass
class OptResult:
    x: np.ndarray                    # найденная точка
    f_val: float                     # значение функции
    n_iter: int                      # число итераций
    n_f: int                         # число вызовов f
    n_grad: int                      # число вызовов grad
    converged: bool                  # сошлось?
    trajectory: list = field(default_factory=list)  # траектория

    def __str__(self):
        status = "✓" if self.converged else "✗"
        return (f"{status} iter={self.n_iter:5d}  f_calls={self.n_f:6d}  "
                f"grad_calls={self.n_grad:5d}  f(x*)={self.f_val:.4e}  "
                f"x*={np.round(self.x, 6)}")


# ─────────────────────────────────────────────────────────────
#  Тестовые функции
# ─────────────────────────────────────────────────────────────

class QuadraticWellConditioned:
    """Хорошо обусловленная квадратичная функция: f(x,y) = x² + y²  (κ=1)"""
    name = "Quadratic (well-cond, κ=1)"
    A = np.array([[2.0, 0.0], [0.0, 2.0]])
    x_opt = np.array([0.0, 0.0])

    def f(self, x: np.ndarray) -> float:
        return float(x @ self.A @ x / 2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def hessian(self) -> np.ndarray:
        return self.A


class QuadraticIllConditioned:
    """Плохо обусловленная квадратичная функция: f(x,y) = 50x² + 0.5y²  (κ=100)"""
    name = "Quadratic (ill-cond, κ=100)"
    A = np.array([[100.0, 0.0], [0.0, 1.0]])
    x_opt = np.array([0.0, 0.0])

    def f(self, x: np.ndarray) -> float:
        return float(x @ self.A @ x / 2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def hessian(self) -> np.ndarray:
        return self.A


class Rosenbrock:
    """Функция Розенброка: f(x,y) = (1-x)² + 100(y-x²)²"""
    name = "Rosenbrock"
    x_opt = np.array([1.0, 1.0])

    def f(self, x: np.ndarray) -> float:
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad(self, x: np.ndarray) -> np.ndarray:
        dfdx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dfdy = 200*(x[1] - x[0]**2)
        return np.array([dfdx, dfdy])


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


class Himmelblau:
    """Функция Химмельблау: (x²+y-11)² + (x+y²-7)²  — 4 минимума"""
    name = "Himmelblau"
    # Четыре минимума:
    x_opt = np.array([3.0, 2.0])

    def f(self, x: np.ndarray) -> float:
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def grad(self, x: np.ndarray) -> np.ndarray:
        dfdx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
        dfdy = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
        return np.array([dfdx, dfdy])


# ─────────────────────────────────────────────────────────────
#  Оптимизаторы
# ─────────────────────────────────────────────────────────────

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
        cf(x)  # один вызов f для контроля значения

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)


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

        # Дробление шага
        while cf(x - alpha * g) > fk - c1 * alpha * np.dot(g, g):
            alpha *= beta
            if alpha < 1e-16:
                break

        x = x - alpha * g
        traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)


def gradient_descent_wolfe(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER
) -> OptResult:
    """Градиентный спуск с дроблением шага (сильные условия Вольфе).
    Используем scipy.optimize.line_search для поиска шага.
    """
    from scipy.optimize import line_search

    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        d = -g  # направление спуска

        # scipy line_search реализует условия Вольфе
        alpha_res = line_search(
            lambda z: cf(z), lambda z: cg(z),
            x, d, gfk=g, c1=c1, c2=c2
        )
        alpha = alpha_res[0]

        if alpha is None or alpha < 1e-16:
            alpha = 1e-4  # fallback

        x = x + alpha * d
        traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)


def gradient_descent_steepest(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    A: Optional[np.ndarray] = None,   # для квадратичных — аналитически
    eps: float = 1e-8,
    max_iter: int = MAX_ITER
) -> OptResult:
    """Метод наискорейшего градиентного спуска.
    Если A задана — используем аналитическую формулу для квадратичных функций.
    Иначе — одномерная минимизация через scipy.
    """
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        if A is not None:
            # Аналитически для квадратичной: α* = ‖g‖²/(gᵀAg)
            denom = g @ A @ g
            alpha = (g @ g) / denom if denom > 1e-16 else 1e-4
        else:
            # Численная одномерная минимизация
            def phi(a):
                val = cf(x - a * g)
                return val
            res = minimize_scalar(phi, bounds=(0, 10), method='bounded')
            alpha = res.x

        x = x - alpha * g
        traj.append(x.copy())

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)


# ─────────────────────────────────────────────────────────────
#  Визуализация
# ─────────────────────────────────────────────────────────────

def plot_contour_with_trajectory(
    f, title, trajectories: dict,
    xlim=(-3, 3), ylim=(-3, 3),
    n_levels=30, figsize=(8, 6), filename=None
):
    """Линии уровня + траектории нескольких методов."""
    xx, yy = np.meshgrid(
        np.linspace(*xlim, 400),
        np.linspace(*ylim, 400)
    )
    zz = np.array([[f(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
                   for row_x, row_y in zip(xx, yy)])

    fig, ax = plt.subplots(figsize=figsize)
    levels = np.percentile(zz[np.isfinite(zz)], np.linspace(0, 95, n_levels))
    cs = ax.contourf(xx, yy, zz, levels=levels, cmap='viridis', alpha=0.7)
    ax.contour(xx, yy, zz, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax)

    colors = ['red', 'orange', 'cyan', 'magenta', 'lime']
    for (label, traj), color in zip(trajectories.items(), colors):
        pts = np.array(traj)
        if len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], '-o', color=color,
                    label=label, markersize=2, linewidth=1.5)
            ax.plot(pts[0, 0], pts[0, 1], 's', color=color, markersize=8)
            ax.plot(pts[-1, 0], pts[-1, 1], '*', color=color, markersize=10)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def plot_iter_vs_step(steps, iters, title, xlabel='Шаг α', filename=None):
    """График числа итераций от шага."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, iters, 'o-', color='steelblue', linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel); ax.set_ylabel('Число итераций (log)')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def plot_iter_vs_eps(epsilons, results_dict, title, filename=None):
    """График iter/f_calls/grad_calls от точности ε."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels_map = ['Итерации', 'Вызовы f', 'Вызовы grad']
    attrs = ['n_iter', 'n_f', 'n_grad']
    colors = ['steelblue', 'darkorange', 'green', 'red']

    for idx, (attr, lbl) in enumerate(zip(attrs, labels_map)):
        ax = axes[idx]
        for (name, results), color in zip(results_dict.items(), colors):
            vals = [getattr(r, attr) for r in results]
            ax.loglog(epsilons, vals, 'o-', label=name, color=color, linewidth=2)
        ax.set_title(f'{title}: {lbl}')
        ax.set_xlabel('ε'); ax.set_ylabel(lbl + ' (log)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────
#  Таблицы
# ─────────────────────────────────────────────────────────────

def print_table(headers, rows, title=""):
    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
             for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  ".join("─"*w for w in col_w))
    for row in rows:
        print(fmt.format(*row))
    print()


# ─────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 1: Постоянный шаг на квадратичных функциях
# ─────────────────────────────────────────────────────────────

def task1_constant_step(eps=1e-8):
    print("\n" + "="*70)
    print("ЗАДАНИЕ 1: Постоянный шаг на квадратичных функциях")
    print("="*70)

    funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0 = np.array([2.0, 2.0])

    # Для хорошо обусловленной: L = max eigenvalue of A = 2  → α < 2/L = 1
    # Для плохо обусловленной: L = 100 → α < 2/100 = 0.02
    steps_per_func = [
        np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.99]),
        np.array([0.0001, 0.001, 0.005, 0.01, 0.015, 0.019, 0.0195, 0.0199]),
    ]

    for func, steps in zip(funcs, steps_per_func):
        rows = []
        iters_list = []
        for alpha in steps:
            r = gradient_descent_const(func.f, func.grad, x0, alpha, eps)
            iters_list.append(r.n_iter if r.converged else None)
            rows.append([
                f"{alpha:.4f}",
                r.n_iter if r.converged else "—",
                f"{r.f_val:.2e}",
                "✓" if r.converged else "✗"
            ])
        print_table(
            ["Шаг α", "Итерации", "f(x*)", "Сошлось?"],
            rows, title=func.name
        )

        # График
        valid = [(s, it) for s, it in zip(steps, iters_list) if it is not None]
        if valid:
            ss, ii = zip(*valid)
            plot_iter_vs_step(ss, ii,
                f"Итерации от шага: {func.name}",
                filename=f"fig_iter_vs_step_{func.name[:4]}.png")

    # Линии уровня для оптимального шага
    for func, steps in zip(funcs, steps_per_func):
        # Найдём оптимальный шаг (наименьшее число итераций)
        best_alpha, best_iter = None, np.inf
        for alpha in steps:
            r = gradient_descent_const(func.f, func.grad, x0, alpha, eps)
            if r.converged and r.n_iter < best_iter:
                best_iter = r.n_iter
                best_alpha = alpha
        if best_alpha is not None:
            r = gradient_descent_const(func.f, func.grad, x0, best_alpha, eps)
            plot_contour_with_trajectory(
                func.f, f"Траектория: {func.name}\n(α*={best_alpha})",
                {f"GD α={best_alpha}": r.trajectory},
                xlim=(-3,3), ylim=(-3,3),
                filename=f"fig_contour_const_{'well' if 'well' in func.name else 'ill'}.png"
            )


# ─────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 2: Постоянный шаг на сложных функциях
# ─────────────────────────────────────────────────────────────

def task2_complex_const_step(eps=1e-8):
    print("\n" + "="*70)
    print("ЗАДАНИЕ 2: Постоянный шаг на сложных функциях")
    print("="*70)

    funcs = [Rosenbrock(), Ackley(), Himmelblau()]
    x0s = {
        "Rosenbrock":  np.array([-1.0, 1.0]),
        "Ackley":      np.array([1.5, 1.0]),
        "Himmelblau":  np.array([0.0, 0.0]),
    }
    steps = [0.1, 0.01, 0.001]

    for func in funcs:
        x0 = x0s[func.name]
        rows = []
        trajectories = {}
        for alpha in steps:
            r = gradient_descent_const(func.f, func.grad, x0, alpha, eps)
            rows.append([
                alpha,
                r.n_iter if r.converged else "—",
                f"{r.f_val:.4e}",
                f"{np.round(r.x, 4)}",
                "✓" if r.converged else "✗"
            ])
            trajectories[f"α={alpha}"] = r.trajectory

        print_table(
            ["Шаг α", "Итерации", "f(x*)", "x*", "Сошлось?"],
            rows, title=func.name
        )

    # Линии уровня для сложных функций
    xlims = {"Rosenbrock": (-2, 2), "Ackley": (-3, 3), "Himmelblau": (-5, 5)}
    ylims = {"Rosenbrock": (-1, 3), "Ackley": (-3, 3), "Himmelblau": (-5, 5)}

    for func in funcs:
        x0 = x0s[func.name]
        trajectories = {}
        for alpha in steps:
            r = gradient_descent_const(func.f, func.grad, x0, alpha, eps)
            trajectories[f"α={alpha}"] = r.trajectory
        plot_contour_with_trajectory(
            func.f, f"Траектории: {func.name}",
            trajectories,
            xlim=xlims[func.name], ylim=ylims[func.name],
            filename=f"fig_contour_complex_{func.name[:3]}.png"
        )


# ─────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 4: Методы дробления шага — зависимость от точности
# ─────────────────────────────────────────────────────────────

def task4_line_search_vs_eps():
    print("\n" + "="*70)
    print("ЗАДАНИЕ 4: Дробление шага — зависимость от точности")
    print("="*70)

    funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0 = np.array([2.0, 2.0])
    epsilons = [10**(-k) for k in range(1, 9)]

    for func in funcs:
        results_by_method = {}

        for method_name, optimizer in [
            ("Армихо",  lambda f, g, x, e: gradient_descent_armijo(f, g, x, eps=e)),
            ("Вольфе",  lambda f, g, x, e: gradient_descent_wolfe(f, g, x, eps=e)),
        ]:
            results = []
            rows = []
            for eps in epsilons:
                r = optimizer(func.f, func.grad, x0, eps)
                results.append(r)
                rows.append([f"{eps:.0e}", r.n_iter, r.n_f, r.n_grad,
                              f"{r.f_val:.2e}", "✓" if r.converged else "✗"])

            print_table(
                ["ε", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"],
                rows, title=f"{func.name} | {method_name}"
            )
            results_by_method[method_name] = results

        suffix = "well" if "well" in func.name else "ill"
        plot_iter_vs_eps(
            epsilons, results_by_method,
            func.name,
            filename=f"fig_eps_{suffix}.png"
        )

        # Линии уровня при eps=1e-4
        eps_fixed = 1e-4
        trajectories = {}
        for method_name, optimizer in [
            ("Армихо",  lambda f, g, x, e: gradient_descent_armijo(f, g, x, eps=e)),
            ("Вольфе",  lambda f, g, x, e: gradient_descent_wolfe(f, g, x, eps=e)),
        ]:
            r = optimizer(func.f, func.grad, x0, eps_fixed)
            trajectories[method_name] = r.trajectory
        plot_contour_with_trajectory(
            func.f, f"Траектории при ε=1e-4: {func.name}",
            trajectories,
            filename=f"fig_traj_linesearch_{func.name[:4]}.png"
        )


# ─────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 5: Дробление шага на сложных функциях
# ─────────────────────────────────────────────────────────────

def task5_complex_line_search(eps=1e-8):
    print("\n" + "="*70)
    print("ЗАДАНИЕ 5: Методы дробления шага на сложных функциях")
    print("="*70)

    funcs = [Rosenbrock(), Himmelblau()]
    start_points = {
        "Rosenbrock": [np.array([-1.0, 1.0]), np.array([0.5, -0.5]), np.array([-1.5, 2.0])],
        "Himmelblau": [np.array([0.0, 0.0]),  np.array([-3.0, 3.0]), np.array([3.0, -1.0])],
    }
    xlims = {"Rosenbrock": (-2, 2), "Himmelblau": (-5, 5)}
    ylims = {"Rosenbrock": (-1, 3), "Himmelblau": (-5, 5)}

    for func in funcs:
        rows = []
        trajectories = {}
        for x0 in start_points[func.name]:
            for method_name, optimizer in [
                ("Армихо",  lambda f, g, x: gradient_descent_armijo(f, g, x, eps=eps)),
                ("Вольфе",  lambda f, g, x: gradient_descent_wolfe(f, g, x, eps=eps)),
            ]:
                r = optimizer(func.f, func.grad, x0)
                key = f"{method_name} x0={np.round(x0,1)}"
                trajectories[key] = r.trajectory
                rows.append([
                    method_name, str(np.round(x0, 2)),
                    r.n_iter, r.n_f, r.n_grad,
                    f"{r.f_val:.4e}", "✓" if r.converged else "✗"
                ])

        print_table(
            ["Метод", "x0", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"],
            rows, title=func.name
        )
        plot_contour_with_trajectory(
            func.f, f"Дробление шага: {func.name}",
            trajectories,
            xlim=xlims[func.name], ylim=ylims[func.name],
            filename=f"fig_complex_linesearch_{func.name[:3]}.png"
        )


# ─────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 6: Метод наискорейшего спуска
# ─────────────────────────────────────────────────────────────

def task6_steepest_descent():
    print("\n" + "="*70)
    print("ЗАДАНИЕ 6: Метод наискорейшего градиентного спуска")
    print("="*70)

    quad_funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0 = np.array([2.0, 2.0])
    epsilons = [10**(-k) for k in range(1, 9)]

    # На квадратичных — зависимость от точности
    for func in quad_funcs:
        rows = []
        results = []
        for eps in epsilons:
            r = gradient_descent_steepest(func.f, func.grad, x0, A=func.A, eps=eps)
            results.append(r)
            rows.append([f"{eps:.0e}", r.n_iter, r.n_f, r.n_grad,
                          f"{r.f_val:.2e}", "✓" if r.converged else "✗"])
        print_table(
            ["ε", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"],
            rows, title=f"Наискорейший спуск: {func.name}"
        )

    # Линии уровня для квадратичных
    for func in quad_funcs:
        r = gradient_descent_steepest(func.f, func.grad, x0, A=func.A)
        suffix = "well" if "well" in func.name else "ill"
        plot_contour_with_trajectory(
            func.f, f"Наискорейший спуск: {func.name}",
            {f"Steepest": r.trajectory},
            filename = f"fig_steepest_{suffix}.png"
        )

    # На сложных функциях
    complex_funcs = [Rosenbrock(), Himmelblau()]
    start_points = {
        "Rosenbrock": [np.array([-1.0, 1.0]), np.array([0.5, -0.5])],
        "Himmelblau": [np.array([0.0, 0.0]),  np.array([-3.0, 3.0])],
    }
    xlims = {"Rosenbrock": (-2, 2), "Himmelblau": (-5, 5)}
    ylims = {"Rosenbrock": (-1, 3), "Himmelblau": (-5, 5)}

    for func in complex_funcs:
        rows = []
        trajectories = {}
        for x0 in start_points[func.name]:
            r = gradient_descent_steepest(func.f, func.grad, x0, eps=1e-8)
            key = f"x0={np.round(x0, 1)}"
            trajectories[key] = r.trajectory
            rows.append([str(np.round(x0, 2)), r.n_iter, r.n_f, r.n_grad,
                          f"{r.f_val:.4e}", "✓" if r.converged else "✗"])

        print_table(
            ["x0", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"],
            rows, title=f"Наискорейший: {func.name}"
        )
        plot_iter_vs_eps(
            epsilons,
            {"Steepest descent": results},
            f"Наискорейший спуск: {func.name}",
            filename=f"fig_eps_steepest_{'well' if 'well' in func.name else 'ill'}.png"
        )
        plot_contour_with_trajectory(
            func.f, f"Наискорейший спуск: {func.name}",
            trajectories,
            xlim=xlims[func.name], ylim=ylims[func.name],
            filename=f"fig_steepest_complex_{func.name[:3]}.png"
        )


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Лабораторная работа 2: Градиентный спуск — классика")
    print("Методы оптимизации (КТ'26)\n")

    task1_constant_step()
    task2_complex_const_step()
    task4_line_search_vs_eps()
    task5_complex_line_search()
    task6_steepest_descent()

    print("\nГотово! Все графики сохранены в PDF.")