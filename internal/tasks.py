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
from internal.utils.visualization import (
    plot_contour_with_trajectory,
    plot_iter_vs_step,
    plot_iter_vs_eps,
    plot_surface,
    print_table,
)


# ══════════════════════════════════════════════════════════════
#  Вспомогательная функция: уникальный суффикс для имён файлов
# ══════════════════════════════════════════════════════════════

def _tag(func) -> str:
    """'well', 'ill', 'Ros', 'Ack', 'Him' — уникальный тег функции."""
    if 'well' in func.name: return 'well'
    if 'ill'  in func.name: return 'ill'
    return func.name[:3]   # Ros / Ack / Him


# ══════════════════════════════════════════════════════════════
#  3D-поверхности всех тестовых функций
# ══════════════════════════════════════════════════════════════

def generate_surfaces():
    """Генерирует файлы fig_surface_*.png."""
    print("\n" + "="*70)
    print("Поверхности тестовых функций")
    print("="*70)

    items = [
        (QuadraticWellConditioned().f,
         'f(x,y) = x²+y²  (κ=1)',
         (-3, 3), (-3, 3), (0, 18),  'fig_surface_well.png', 'plasma'),

        (QuadraticIllConditioned().f,
         'g(x,y) = 50x²+0.5y²  (κ=100)',
         (-3, 3), (-3, 3), (0, 400), 'fig_surface_ill.png',  'plasma'),

        (Rosenbrock().f,
         'Розенброк: (1-x)²+100(y-x²)²',
         (-2, 2), (-1, 3), (0, 800), 'fig_surface_Ros.png',  'inferno'),

        (Ackley().f,
         'Экли',
         (-4, 4), (-4, 4), None,     'fig_surface_Ack.png',  'coolwarm'),

        (Himmelblau().f,
         'Химмельблау: (x²+y-11)²+(x+y²-7)²',
         (-5, 5), (-5, 5), (0, 500), 'fig_surface_Him.png',  'turbo'),
    ]
    for args in items:
        plot_surface(*args)


# ══════════════════════════════════════════════════════════════
#  ЗАДАНИЕ 1: Постоянный шаг на квадратичных функциях
# ══════════════════════════════════════════════════════════════
#
#  Генерируемые файлы:
#    fig_iter_vs_step_well.png   fig_iter_vs_step_ill.png
#    fig_contour_const_well.png  fig_contour_const_ill.png

def constant_step_quadratics(eps: float = 1e-8):
    print("\n" + "="*70)
    print("ЗАДАНИЕ 1: Постоянный шаг на квадратичных функциях")
    print("="*70)

    funcs = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0    = np.array([2.0, 2.0])
    steps_per_func = [
        np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.99]),
        np.array([0.0001, 0.001, 0.005, 0.01, 0.015, 0.019, 0.0195, 0.0199]),
    ]

    for func, steps in zip(funcs, steps_per_func):
        tag  = _tag(func)
        rows = []
        sv   = []   # шаги для которых сошлось
        iv   = []   # число итераций

        for alpha in steps:
            r = gradient_descent_const(func.f, func.grad, x0, alpha, eps)
            rows.append([
                f"{alpha:.4f}",
                r.n_iter if r.converged else "—",
                f"{r.f_val:.2e}",
                "✓" if r.converged else "✗",
            ])
            if r.converged:
                sv.append(alpha)
                iv.append(r.n_iter)

        print_table(["Шаг α", "Итерации", "f(x*)", "Сошлось?"],
                    rows, title=func.name)

        if sv:
            plot_iter_vs_step(
                sv, iv,
                f"Итерации от шага: {func.name}",
                filename=f"fig_iter_vs_step_{tag}.png",
            )

        # Контурный график с оптимальным шагом
        best_alpha = min(sv, key=lambda a:
                         gradient_descent_const(func.f, func.grad, x0, a, eps).n_iter)
        r_best = gradient_descent_const(func.f, func.grad, x0, best_alpha, eps)
        plot_contour_with_trajectory(
            func.f,
            f"Постоянный шаг: {func.name}\n(α*={best_alpha})",
            {f"GD α={best_alpha}": r_best.trajectory},
            xlim=(-3, 3), ylim=(-3, 3),
            filename=f"fig_contour_const_{tag}.png",
        )


# ══════════════════════════════════════════════════════════════
#  ЗАДАНИЕ 2: Постоянный шаг на сложных функциях
# ══════════════════════════════════════════════════════════════
#
#  Генерируемые файлы:
#    fig_contour_complex_Ros.png
#    fig_contour_complex_Ack.png
#    fig_contour_complex_Him.png

def constant_step_complex(eps: float = 1e-8):
    print("\n" + "="*70)
    print("ЗАДАНИЕ 2: Постоянный шаг на сложных функциях")
    print("="*70)

    configs = [
        (Rosenbrock(),  np.array([-1.0, 1.0]), (-2, 2), (-1, 3)),
        (Ackley(),      np.array([1.5,  1.0]), (-3, 3), (-3, 3)),
        (Himmelblau(),  np.array([0.0,  0.0]), (-5, 5), (-5, 5)),
    ]
    steps = [0.1, 0.01, 0.001]

    for func, x0, xlim, ylim in configs:
        rows  = []
        trajs = {}

        for alpha in steps:
            r = gradient_descent_const(func.f, func.grad, x0, alpha, eps)
            rows.append([
                alpha,
                r.n_iter if r.converged else "—",
                f"{r.f_val:.4e}",
                str(np.round(r.x, 4)),
                "✓" if r.converged else "✗",
            ])
            trajs[f"α={alpha}"] = r.trajectory

        print_table(["Шаг α", "Итерации", "f(x*)", "x*", "Сошлось?"],
                    rows, title=func.name)

        plot_contour_with_trajectory(
            func.f,
            f"Постоянный шаг: {func.name}",
            trajs, xlim, ylim,
            filename=f"fig_contour_complex_{_tag(func)}.png",
        )


# ══════════════════════════════════════════════════════════════
#  ЗАДАНИЕ 4: Дробление шага — зависимость от точности ε
# ══════════════════════════════════════════════════════════════
#
#  Генерируемые файлы:
#    fig_eps_well.png              fig_eps_ill.png
#    fig_traj_linesearch_well.png  fig_traj_linesearch_ill.png

def line_search_precision_dependency():
    print("\n" + "="*70)
    print("ЗАДАНИЕ 4: Дробление шага — зависимость от точности")
    print("="*70)

    funcs    = [QuadraticWellConditioned(), QuadraticIllConditioned()]
    x0       = np.array([2.0, 2.0])
    epsilons = [10**(-k) for k in range(1, 9)]

    for func in funcs:
        tag = _tag(func)
        results_by_method = {}

        for method_name, method_fn in [
            ("Армихо", gradient_descent_armijo),
            ("Вольфе", gradient_descent_wolfe),
        ]:
            rows = []
            rs   = []
            for ev in epsilons:
                r = method_fn(func.f, func.grad, x0, eps=ev)
                rs.append(r)
                rows.append([
                    f"{ev:.0e}", r.n_iter, r.n_f, r.n_grad,
                    f"{r.f_val:.2e}", "✓" if r.converged else "✗",
                ])
            print_table(
                ["ε", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"],
                rows, title=f"{func.name} | {method_name}",
            )
            results_by_method[method_name] = rs

        plot_iter_vs_eps(
            epsilons, results_by_method, func.name,
            filename=f"fig_eps_{tag}.png",
        )

        # Контурный график при ε = 1e-4
        trajs = {
            name: method_fn(func.f, func.grad, x0, eps=1e-4).trajectory
            for name, method_fn in [
                ("Армихо", gradient_descent_armijo),
                ("Вольфе", gradient_descent_wolfe),
            ]
        }
        plot_contour_with_trajectory(
            func.f,
            f"Траектории при ε=1e-4: {func.name}",
            trajs, (-3, 3), (-3, 3),
            filename=f"fig_traj_linesearch_{tag}.png",
        )


# ══════════════════════════════════════════════════════════════
#  ЗАДАНИЕ 5: Дробление шага на сложных функциях
# ══════════════════════════════════════════════════════════════
#
#  Генерируемые файлы:
#    fig_complex_linesearch_Ros.png
#    fig_complex_linesearch_Him.png

def line_search_complex(eps: float = 1e-8):
    print("\n" + "="*70)
    print("ЗАДАНИЕ 5: Дробление шага на сложных функциях")
    print("="*70)

    configs = [
        (Rosenbrock(),
         [np.array([-1.0, 1.0]), np.array([0.5, -0.5]), np.array([-1.5, 2.0])],
         (-2, 2), (-1, 3)),
        (Himmelblau(),
         [np.array([0.0, 0.0]), np.array([-3.0, 3.0]), np.array([3.0, -1.0])],
         (-5, 5), (-5, 5)),
    ]

    for func, starts, xlim, ylim in configs:
        rows  = []
        trajs = {}

        for x0 in starts:
            for method_name, method_fn in [
                ("Армихо", gradient_descent_armijo),
                ("Вольфе", gradient_descent_wolfe),
            ]:
                r   = method_fn(func.f, func.grad, x0, eps=eps)
                key = f"{method_name} x0={np.round(x0, 1)}"
                trajs[key] = r.trajectory
                rows.append([
                    method_name,
                    str(np.round(x0, 2)),
                    r.n_iter if r.converged else "—",
                    r.n_f, r.n_grad,
                    f"{r.f_val:.4e}",
                    "✓" if r.converged else "✗",
                ])

        print_table(
            ["Метод", "x0", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"],
            rows, title=func.name,
        )
        plot_contour_with_trajectory(
            func.f,
            f"Дробление шага: {func.name}",
            trajs, xlim, ylim,
            filename=f"fig_complex_linesearch_{_tag(func)}.png",
        )


# ══════════════════════════════════════════════════════════════
#  ЗАДАНИЕ 6: Метод наискорейшего градиентного спуска
# ══════════════════════════════════════════════════════════════
#
#  Генерируемые файлы:
#    fig_steepest_well.png         fig_steepest_ill.png
#    fig_eps_steepest_ill.png
#    fig_steepest_complex_Ros.png  fig_steepest_complex_Him.png

def steepest_descent_analysis():
    print("\n" + "="*70)
    print("ЗАДАНИЕ 6: Метод наискорейшего градиентного спуска")
    print("="*70)

    x0       = np.array([2.0, 2.0])
    epsilons = [10**(-k) for k in range(1, 9)]

    # Квадратичные функции — зависимость от точности
    for func in [QuadraticWellConditioned(), QuadraticIllConditioned()]:
        tag  = _tag(func)
        rows = []
        rs   = []
        for ev in epsilons:
            r = gradient_descent_steepest(func.f, func.grad, x0, A=func.A, eps=ev)
            rs.append(r)
            rows.append([
                f"{ev:.0e}", r.n_iter, r.n_f, r.n_grad,
                f"{r.f_val:.2e}", "✓" if r.converged else "✗",
            ])
        print_table(
            ["ε", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"],
            rows, title=f"Наискорейший спуск: {func.name}",
        )

        # График от ε только для плохо обусловленной (интересен)
        if tag == 'ill':
            plot_iter_vs_eps(
                epsilons, {"Наискорейший": rs}, func.name,
                filename="fig_eps_steepest_ill.png",
            )

        # Контурный график
        r_plot = gradient_descent_steepest(func.f, func.grad, x0, A=func.A, eps=1e-8)
        plot_contour_with_trajectory(
            func.f,
            f"Наискорейший спуск: {func.name}",
            {"Наискорейший": r_plot.trajectory},
            xlim=(-3, 3), ylim=(-3, 3),
            filename=f"fig_steepest_{tag}.png",
        )

    # Сложные функции
    configs = [
        (Rosenbrock(),
         [np.array([-1.0, 1.0]), np.array([0.5, -0.5])],
         (-2, 2), (-1, 3)),
        (Himmelblau(),
         [np.array([0.0, 0.0]), np.array([-3.0, 3.0])],
         (-5, 5), (-5, 5)),
    ]

    for func, starts, xlim, ylim in configs:
        rows  = []
        trajs = {}

        for x0 in starts:
            r   = gradient_descent_steepest(func.f, func.grad, x0, eps=1e-8)
            key = f"x0={np.round(x0, 1)}"
            trajs[key] = r.trajectory
            rows.append([
                str(np.round(x0, 2)),
                r.n_iter if r.converged else "—",
                r.n_f, r.n_grad,
                f"{r.f_val:.4e}",
                "✓" if r.converged else "✗",
            ])

        print_table(
            ["x0", "Итерации", "f_calls", "grad_calls", "f(x*)", "Сошлось?"],
            rows, title=f"Наискорейший: {func.name}",
        )
        plot_contour_with_trajectory(
            func.f,
            f"Наискорейший спуск: {func.name}",
            trajs, xlim, ylim,
            filename=f"fig_steepest_complex_{_tag(func)}.png",
        )