import numpy as np

from internal.functions import (
    QuadraticWellConditioned,
    QuadraticIllConditioned,
    Rosenbrock,
    Ackley,
    Himmelblau,
)
from internal.optimizers import (
    gradient_descent_const,
    gradient_descent_armijo,
    gradient_descent_wolfe,
    gradient_descent_steepest,
)
from internal.utils import (
    plot_contour_with_trajectory,
    plot_iter_vs_step,
    plot_iter_vs_eps,
    print_table,
)
from internal.runner import run


def constant_step_quadratics(eps=1e-8):
    print("\n" + "="*70)
    print("Постоянный шаг на квадратичных функциях")
    print("="*70)

    funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0 = np.array([2.0, 2.0])

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
                "✓" if r.converged else "✗",
            ])
        print_table(["Шаг α", "Итерации", "f(x*)", "Сошлось?"], rows, title=func.name)

        valid = [(s, it) for s, it in zip(steps, iters_list) if it is not None]
        if valid:
            ss, ii = zip(*valid)
            plot_iter_vs_step(ss, ii, f"Итерации от шага: {func.name}", filename=f"fig_iter_vs_step_{func.name[:4]}.png")

    for func, steps in zip(funcs, steps_per_func):
        best_alpha, best_iter = None, np.inf
        for alpha in steps:
            r = gradient_descent_const(func.f, func.grad, x0, alpha, eps)
            if r.converged and r.n_iter < best_iter:
                best_iter = r.n_iter
                best_alpha = alpha
        if best_alpha is not None:
            r = gradient_descent_const(func.f, func.grad, x0, best_alpha, eps)
            plot_contour_with_trajectory(
                func.f,
                f"Траектория: {func.name}\n(α*={best_alpha})",
                {f"GD α={best_alpha}": r.trajectory},
                xlim=(-3, 3),
                ylim=(-3, 3),
                filename=f"fig_contour_const_{'well' if 'well' in func.name else 'ill'}.png",
            )


def constant_step_complex(eps=1e-8):
    print("\n" + "="*70)
    print("Постоянный шаг на сложных функциях")
    print("="*70)

    funcs = [Rosenbrock(), Ackley(), Himmelblau()]
    x0s = {
        "Rosenbrock": np.array([-1.0, 1.0]),
        "Ackley": np.array([1.5, 1.0]),
        "Himmelblau": np.array([0.0, 0.0]),
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
                "✓" if r.converged else "✗",
            ])
            trajectories[f"α={alpha}"] = r.trajectory

        print_table(["Шаг α", "Итерации", "f(x*)", "x*", "Сошлось?"], rows, title=func.name)

    xlims = {"Rosenbrock": (-2, 2), "Ackley": (-3, 3), "Himmelblau": (-5, 5)}
    ylims = {"Rosenbrock": (-1, 3), "Ackley": (-3, 3), "Himmelblau": (-5, 5)}

    for func in funcs:
        x0 = x0s[func.name]
        trajectories = {}
        for alpha in steps:
            r = gradient_descent_const(func.f, func.grad, x0, alpha, eps)
            trajectories[f"α={alpha}"] = r.trajectory
        plot_contour_with_trajectory(
            func.f,
            f"Траектории: {func.name}",
            trajectories,
            xlim=xlims[func.name],
            ylim=ylims[func.name],
            filename=f"fig_contour_complex_{func.name[:3]}.png",
        )


def line_search_precision_dependency():
    print("\n" + "="*70)
    print("Дробление шага — зависимость от точности")
    print("="*70)

    funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0 = np.array([2.0, 2.0])
    epsilons = [10**(-k) for k in range(1, 9)]

    for func in funcs:
        results_by_method = {}

        for method_name, optimizer in [
            ("Армихо", lambda f, g, x, e: gradient_descent_armijo(f, g, x, eps=e)),
            ("Вольфе", lambda f, g, x, e: gradient_descent_wolfe(f, g, x, eps=e)),
        ]:
            results = []
            rows = []
            for eps in epsilons:
                r = optimizer(func.f, func.grad, x0, eps)
                results.append(r)
                rows.append([f"{eps:.0e}", r.n_iter, r.n_f, r.n_grad, f"{r.f_val:.2e}", "✓" if r.converged else "✗"])

            print_table([
                "ε",
                "Итерации",
                "f_calls",
                "grad_calls",
                "f(x*)",
                "Сошлось?",
            ], rows, title=f"{func.name} | {method_name}")

            results_by_method[method_name] = results

        suffix = "well" if "well" in func.name else "ill"
        plot_iter_vs_eps(epsilons, results_by_method, func.name, filename=f"fig_eps_{suffix}.png")

        eps_fixed = 1e-4
        trajectories = {}
        for method_name, optimizer in [
            ("Армихо", lambda f, g, x, e: gradient_descent_armijo(f, g, x, eps=e)),
            ("Вольфе", lambda f, g, x, e: gradient_descent_wolfe(f, g, x, eps=e)),
        ]:
            r = optimizer(func.f, func.grad, x0, eps_fixed)
            trajectories[method_name] = r.trajectory
        plot_contour_with_trajectory(
            func.f,
            f"Траектории при ε=1e-4: {func.name}",
            trajectories,
            filename=f"fig_traj_linesearch_{func.name[:4]}.png",
        )


def line_search_complex(eps=1e-8):
    print("\n" + "="*70)
    print("Дробление шага на сложных функциях")
    print("="*70)

    funcs = [Rosenbrock(), Himmelblau()]
    start_points = {
        "Rosenbrock": [np.array([-1.0, 1.0]), np.array([0.5, -0.5]), np.array([-1.5, 2.0])],
        "Himmelblau": [np.array([0.0, 0.0]), np.array([-3.0, 3.0]), np.array([3.0, -1.0])],
    }
    xlims = {"Rosenbrock": (-2, 2), "Himmelblau": (-5, 5)}
    ylims = {"Rosenbrock": (-1, 3), "Himmelblau": (-5, 5)}

    for func in funcs:
        rows = []
        trajectories = {}
        for x0 in start_points[func.name]:
            for method_name, optimizer in [
                ("Армихо", lambda f, g, x: gradient_descent_armijo(f, g, x, eps=eps)),
                ("Вольфе", lambda f, g, x: gradient_descent_wolfe(f, g, x, eps=eps)),
            ]:
                r = optimizer(func.f, func.grad, x0)
                key = f"{method_name} x0={np.round(x0,1)}"
                trajectories[key] = r.trajectory
                rows.append([
                    method_name,
                    str(np.round(x0, 2)),
                    r.n_iter,
                    r.n_f,
                    r.n_grad,
                    f"{r.f_val:.4e}",
                    "✓" if r.converged else "✗",
                ])

        print_table([
            "Метод",
            "x0",
            "Итерации",
            "f_calls",
            "grad_calls",
            "f(x*)",
            "Сошлось?",
        ], rows, title=func.name)
        plot_contour_with_trajectory(
            func.f,
            f"Дробление шага: {func.name}",
            trajectories,
            xlim=xlims[func.name],
            ylim=ylims[func.name],
            filename=f"fig_complex_linesearch_{func.name[:3]}.png",
        )


def steepest_descent_analysis():
    print("\n" + "="*70)
    print("Метод наискорейшего градиента")
    print("="*70)

    quad_funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0 = np.array([2.0, 2.0])
    epsilons = [10**(-k) for k in range(1, 9)]

    for func in quad_funcs:
        rows = []
        results = []
        for eps in epsilons:
            r = gradient_descent_steepest(func.f, func.grad, x0, A=func.A, eps=eps)
            results.append(r)
            rows.append([f"{eps:.0e}", r.n_iter, r.n_f, r.n_grad, f"{r.f_val:.2e}", "✓" if r.converged else "✗"])
        print_table(["ε", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"], rows, title=f"Наискорейший спуск: {func.name}")

    for func in quad_funcs:
        r = gradient_descent_steepest(func.f, func.grad, x0, A=func.A)
        suffix = "well" if "well" in func.name else "ill"
        plot_contour_with_trajectory(func.f, f"Наискорейший спуск: {func.name}", {f"Steepest": r.trajectory}, filename=f"fig_steepest_{suffix}.png")

    complex_funcs = [Rosenbrock(), Himmelblau()]
    start_points = {
        "Rosenbrock": [np.array([-1.0, 1.0]), np.array([0.5, -0.5])],
        "Himmelblau": [np.array([0.0, 0.0]), np.array([-3.0, 3.0])],
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
            rows.append([str(np.round(x0, 2)), r.n_iter, r.n_f, r.n_grad, f"{r.f_val:.4e}", "✓" if r.converged else "✗"])

        print_table(["x0", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"], rows, title=f"Наискорейший: {func.name}")
        plot_contour_with_trajectory(
            func.f,
            f"Наискорейший спуск: {func.name}",
            trajectories,
            xlim=xlims[func.name],
            ylim=ylims[func.name],
            filename=f"fig_steepest_complex_{func.name[:3]}.png",
        )
