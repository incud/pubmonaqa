from pathlib import Path

import numpy as np
import pandas as pd

from monaqa2.data.filename import CLASSICAL_QUERY_FILE, QUANTUM_QUERY_FILE
from monaqa2.data.classical_query import get_classical_query_stats, get_classical_query_fit
from monaqa2.data.quantum_query import get_quantum_query_stats, get_quantum_query_fit


MOVES = ("local1", "uniform", "layden")


def _filter_stats_table(table: pd.DataFrame, beta: float, min_count: int) -> pd.DataFrame:
    table = table[np.isclose(table["beta"].astype(float), float(beta)) & (table["count"].astype(int) >= min_count) & np.isfinite(table["n"].astype(float)) & np.isfinite(table["center"].astype(float)) & (table["center"].astype(float) > 0.0)].copy()

    if table.empty:
        return table

    table["n"] = table["n"].astype(float)
    table["center"] = table["center"].astype(float)

    return table.sort_values("n")


def _observed_rows(table: pd.DataFrame, walk: str, move: str, A: float, b: float) -> list[dict]:
    return [{"n": int(row.n), "walk": walk, "move": move, "num_queries": float(row.center), "source": "observed", "A": float(A), "b": float(b)} for row in table.itertuples(index=False)]


def get_classical_quantum_query_vs_n_table(
    beta: float,
    a: int | float,
    q0_mode: str,
    epsilon: float,
    classical_query_file: Path = CLASSICAL_QUERY_FILE,
    quantum_query_file: Path = QUANTUM_QUERY_FILE,
    statistic: str = "mean+std",
    min_count: int = 1,
    n_fit_min: int | None = 5,
    n_fit_max: int | None = 10,
    n_project_values=tuple(range(10, 101, 10)),
    moves: tuple[str, ...] = MOVES,
    only_ok: bool = True,
) -> pd.DataFrame:
    """
    Build a query table with empirical/statistical rows only.

    The returned table contains columns ["n", "walk", "move", "num_queries", "source", "A", "b"]. The coefficients A,b are the fitted parameters of num_queries(n)=A exp(b n) for the corresponding (walk, move). The n_project_values argument is kept only for backward compatibility and is intentionally ignored.
    """
    rows = []

    for move in moves:
        classical_table = get_classical_query_stats(proposal=move, a=a, q0_mode=q0_mode, epsilon=epsilon, in_file=classical_query_file, statistic=statistic, only_ok=only_ok)
        classical_table = _filter_stats_table(classical_table, beta=beta, min_count=min_count)

        try:
            A, b = get_classical_query_fit(proposal=move, a=a, q0_mode=q0_mode, epsilon=epsilon, beta=beta, in_file=classical_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max, only_ok=only_ok)
        except ValueError:
            A, b = np.nan, np.nan

        rows.extend(_observed_rows(classical_table, walk="classical", move=move, A=A, b=b))

        quantum_table = get_quantum_query_stats(proposal=move, a=a, q0_mode=q0_mode, in_file=quantum_query_file, statistic=statistic, only_ok=only_ok)
        quantum_table = _filter_stats_table(quantum_table, beta=beta, min_count=min_count)

        try:
            A, b = get_quantum_query_fit(proposal=move, a=a, q0_mode=q0_mode, beta=beta, in_file=quantum_query_file, statistic=statistic, n_min=n_fit_min, n_max=n_fit_max, only_ok=only_ok)
        except ValueError:
            A, b = np.nan, np.nan

        rows.extend(_observed_rows(quantum_table, walk="quantum", move=move, A=A, b=b))

    return pd.DataFrame(rows, columns=["n", "walk", "move", "num_queries", "source", "A", "b"]).sort_values(["walk", "move", "n"]).reset_index(drop=True)
