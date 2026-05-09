# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
#
#
# def plot_contour_with_trajectory(
#     f, title, trajectories: dict,
#     xlim=(-3, 3), ylim=(-3, 3),
#     n_levels=30, figsize=(8, 6), filename=None
# ):
#     """Линии уровня + траектории нескольких методов."""
#     xx, yy = np.meshgrid(
#         np.linspace(*xlim, 400),
#         np.linspace(*ylim, 400)
#     )
#     zz = np.array([[f(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
#                    for row_x, row_y in zip(xx, yy)])
#
#     fig, ax = plt.subplots(figsize=figsize)
#     levels = np.percentile(zz[np.isfinite(zz)], np.linspace(0, 95, n_levels))
#     cs = ax.contourf(xx, yy, zz, levels=levels, cmap='viridis', alpha=0.7)
#     ax.contour(xx, yy, zz, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
#     plt.colorbar(cs, ax=ax)
#
#     colors = ['red', 'orange', 'cyan', 'magenta', 'lime']
#     for (label, traj), color in zip(trajectories.items(), colors):
#         pts = np.array(traj)
#         if len(pts) > 1:
#             ax.plot(pts[:, 0], pts[:, 1], '-o', color=color,
#                     label=label, markersize=2, linewidth=1.5)
#             ax.plot(pts[0, 0], pts[0, 1], 's', color=color, markersize=8)
#             ax.plot(pts[-1, 0], pts[-1, 1], '*', color=color, markersize=10)
#
#     ax.set_title(title, fontsize=13)
#     ax.set_xlabel('x'); ax.set_ylabel('y')
#     ax.legend(loc='upper right', fontsize=8)
#     ax.set_xlim(xlim); ax.set_ylim(ylim)
#     plt.tight_layout()
#     if filename:
#         from pathlib import Path
#         base = Path(__file__).resolve().parents[2] / 'results'
#         plots_dir = base / 'plots'
#         plots_dir.mkdir(parents=True, exist_ok=True)
#         save_path = plots_dir / filename if not Path(filename).is_absolute() else Path(filename)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.show()
#
#
# def plot_iter_vs_step(steps, iters, title, xlabel='Шаг α', filename=None):
#     """График числа итераций от шага."""
#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.semilogy(steps, iters, 'o-', color='steelblue', linewidth=2)
#     ax.set_title(title, fontsize=13)
#     ax.set_xlabel(xlabel); ax.set_ylabel('Число итераций (log)')
#     ax.grid(True, alpha=0.4)
#     plt.tight_layout()
#     if filename:
#         from pathlib import Path
#         base = Path(__file__).resolve().parents[2] / 'results'
#         plots_dir = base / 'plots'
#         plots_dir.mkdir(parents=True, exist_ok=True)
#         save_path = plots_dir / filename if not Path(filename).is_absolute() else Path(filename)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.show()
#
#
# def plot_iter_vs_eps(epsilons, results_dict, title, filename=None):
#     """График iter/f_calls/grad_calls от точности ε."""
#     fig, axes = plt.subplots(1, 3, figsize=(14, 4))
#     labels_map = ['Итерации', 'Вызовы f', 'Вызовы grad']
#     attrs = ['n_iter', 'n_f', 'n_grad']
#     colors = ['steelblue', 'darkorange', 'green', 'red']
#
#     for idx, (attr, lbl) in enumerate(zip(attrs, labels_map)):
#         ax = axes[idx]
#         for (name, results), color in zip(results_dict.items(), colors):
#             vals = [getattr(r, attr) for r in results]
#             ax.loglog(epsilons, vals, 'o-', label=name, color=color, linewidth=2)
#         ax.set_title(f'{title}: {lbl}')
#         ax.set_xlabel('ε'); ax.set_ylabel(lbl + ' (log)')
#         ax.legend(fontsize=7); ax.grid(True, alpha=0.4)
#
#     plt.tight_layout()
#     if filename:
#         from pathlib import Path
#         base = Path(__file__).resolve().parents[2] / 'results'
#         plots_dir = base / 'plots'
#         plots_dir.mkdir(parents=True, exist_ok=True)
#         save_path = plots_dir / filename if not Path(filename).is_absolute() else Path(filename)
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.show()
#
#
# def print_table(headers, rows, title=""):
#     if title:
#         print(f"\n{'─' * 60}")
#         print(f"  {title}")
#         print(f"{'─' * 60}")
#     col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
#              for i, h in enumerate(headers)]
#     fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
#     print(fmt.format(*headers))
#     print("  ".join("─"*w for w in col_w))
#     for row in rows:
#         print(fmt.format(*row))
#     print()
#     # Сохраняем таблицу в results/tables
#     from pathlib import Path
#     import csv
#     base = Path(__file__).resolve().parents[2] / "results"
#     tables_dir = base / "tables"
#     tables_dir.mkdir(parents=True, exist_ok=True)
#     safe_title = title if title else "table"
#     fname = "".join([c if (c.isalnum() or c in (" ", "_")) else "_" for c in safe_title]).strip().replace(" ", "_") + ".csv"
#     save_path = tables_dir / fname
#     with open(save_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(headers)
#         for r in rows:
#             writer.writerow([str(item) for item in r])


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ─── Куда сохранять графики ───────────────────────────────────
PLOTS_DIR = Path(__file__).resolve().parents[2] / "results" / "plots"


def _save(fig, filename: str):
    if filename:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Линии уровня + траектории ────────────────────────────────

def plot_contour_with_trajectory(
    f, title, trajectories: dict,
    xlim=(-3, 3), ylim=(-3, 3),
    n_levels=30, figsize=(8, 6), filename=None
):
    xx, yy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    zz = np.array([[f(np.array([xi, yi])) for xi, yi in zip(rx, ry)]
                   for rx, ry in zip(xx, yy)])

    fig, ax = plt.subplots(figsize=figsize)
    finite = zz[np.isfinite(zz)]
    levels = np.percentile(finite, np.linspace(0, 95, n_levels))
    cs = ax.contourf(xx, yy, zz, levels=levels, cmap='viridis', alpha=0.7)
    ax.contour(xx, yy, zz, levels=levels, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax)

    colors = ['red', 'orange', 'cyan', 'magenta', 'lime',
              'blue', 'purple', 'brown', 'pink', 'gray']
    for (label, traj), color in zip(trajectories.items(), colors):
        pts = np.array(traj)
        if len(pts) < 2:
            continue
        # обрезаем точки вышедшие за границы
        mask = ((pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1]) &
                (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1]))
        cut = int(np.argmin(mask)) if not mask.all() else len(pts)
        pts = pts[:max(cut, 2)]
        ax.plot(pts[:, 0], pts[:, 1], '-o', color=color,
                label=label, markersize=2, linewidth=1.5)
        ax.plot(pts[0,  0], pts[0,  1], 's', color=color, markersize=8)
        ax.plot(pts[-1, 0], pts[-1, 1], '*', color=color, markersize=10)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(loc='best', fontsize=7, framealpha=0.8)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    plt.tight_layout()
    _save(fig, filename)


# ─── Итерации от шага ─────────────────────────────────────────

def plot_iter_vs_step(steps, iters, title, xlabel='Шаг α', filename=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, iters, 'o-', color='steelblue', linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel); ax.set_ylabel('Число итераций (log)')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, filename)


# ─── iter / f_calls / grad_calls от ε ────────────────────────

def plot_iter_vs_eps(epsilons, results_dict, title, filename=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels_map = ['Итерации', 'Вызовы f', 'Вызовы grad']
    attrs      = ['n_iter',   'n_f',      'n_grad']
    colors     = ['steelblue', 'darkorange', 'green', 'red']

    for ax, attr, lbl in zip(axes, attrs, labels_map):
        for (name, results), color in zip(results_dict.items(), colors):
            vals = [getattr(r, attr) for r in results]
            ax.loglog(epsilons, vals, 'o-', label=name, color=color, linewidth=2)
        ax.set_title(f'{title}: {lbl}')
        ax.set_xlabel('ε'); ax.set_ylabel(lbl + ' (log)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    _save(fig, filename)


# ─── 3D-поверхность ───────────────────────────────────────────

def plot_surface(f, title, xlim, ylim, zlim, filename, cmap='viridis'):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    xx, yy = np.meshgrid(np.linspace(*xlim, 150), np.linspace(*ylim, 150))
    zz = np.vectorize(lambda a, b: f(np.array([a, b])))(xx, yy)
    zz = np.where(np.isfinite(zz), zz, np.nan)
    if zlim:
        zz = np.clip(zz, *zlim)
    fig = plt.figure(figsize=(7, 4.5))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap=cmap, alpha=0.88, lw=0, antialiased=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f')
    plt.tight_layout()
    _save(fig, filename)


# ─── Таблица в консоль (+ CSV) ────────────────────────────────

def print_table(headers, rows, title=""):
    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
    col_w = [max(len(str(h)),
                 max((len(str(r[i])) for r in rows), default=0)) + 2
             for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  ".join("─" * w for w in col_w))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print()

    # сохраняем CSV
    import csv
    tables_dir = Path(__file__).resolve().parents[2] / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in " _" else "_"
                   for c in (title or "table")).strip().replace(" ", "_")
    with open(tables_dir / f"{safe}.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(headers)
        for r in rows:
            w.writerow([str(c) for c in r])