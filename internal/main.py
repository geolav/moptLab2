from internal.tasks import (
    constant_step_quadratics,
    constant_step_complex,
    line_search_precision_dependency,
    line_search_complex,
    steepest_descent_analysis,
)
from internal.runner import run

def main():
    run([
        constant_step_quadratics,
        constant_step_complex,
        line_search_precision_dependency,
        line_search_complex,
        steepest_descent_analysis,
    ], output_root="results", save_graphs=True, save_tables=True)

if __name__ == "__main__":
    main()
