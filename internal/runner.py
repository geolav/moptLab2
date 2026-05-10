from pathlib import Path
from typing import Callable, Sequence


def run(
    tasks: Sequence[Callable],
    output_root: str = "results",
    save_graphs: bool = True,
    save_tables: bool = True,
) -> None:
    root = Path(output_root)
    plots_dir = root / "plots"
    tables_dir = root / "tables"

    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        print("\n" + "#" * 80)
        print(f" Running: {task.__name__} ".center(80))
        print("#" * 80 + "\n")
        task(save_graphs=save_graphs, save_tables=save_tables)

    if save_graphs:
        print(f"Graphs saved in: {plots_dir}")
    if save_tables:
        print(f"Tables saved in: {tables_dir}")
