"""
Bootstrap engine for the Spatial Point Pattern Test (SPPT).

Implements the sparse-matrix multinomial bootstrap algorithm for a single
count variable, faithfully mirroring the R implementation in sppt.R (lines 393-445).

Algorithm:
    1. Expand aggregated counts to individual events (uncount)
    2. Build sparse one-hot matrix  (n × G)
    3. Draw B multinomial bootstrap samples  (n × B weight matrix)
    4. Aggregate via matrix multiply:  group_counts = onehot.T @ W  →  (G × B)
    5. Optionally convert to percentages (column-wise / column_sum * 100)
    6. Extract row-wise quantiles for confidence bounds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def bootstrap_single_variable(
    data: pd.DataFrame,
    group_col: str,
    count_col: str,
    B: int = 200,
    new_col: str | None = None,
    seed: int | None = None,
    conf_level: float = 0.95,
    use_percentages: bool = True,
) -> pd.DataFrame:
    """Run bootstrap resampling on a single count variable and return CI bounds.

    Parameters
    ----------
    data : DataFrame or GeoDataFrame
        Must contain *group_col* and *count_col* columns.
    group_col : str
        Column identifying spatial (or other) groups.
    count_col : str
        Column with integer counts to bootstrap.
    B : int
        Number of bootstrap samples.
    new_col : str or None
        Prefix for the output columns ``{new_col}_L`` / ``{new_col}_U``.
        Defaults to *count_col*.
    seed : int or None
        Random seed for reproducibility.
    conf_level : float
        Confidence level (e.g. 0.95 for 95 %).
    use_percentages : bool
        If True, convert bootstrap counts to percentages (each sample sums to 100).

    Returns
    -------
    DataFrame
        A two-column DataFrame indexed by group value, with columns
        ``{new_col}_L`` and ``{new_col}_U``.  This is meant to be joined
        back onto the original data by the caller.
    """
    if new_col is None:
        new_col = count_col

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # ── 1. Drop geometry (if GeoDataFrame) and select relevant columns ──────
    try:
        df = data.drop(columns="geometry", errors="ignore")
    except Exception:
        df = data

    df = df[[group_col, count_col]].copy()
    df[count_col] = df[count_col].fillna(0).astype(int)

    # Store original group dtype for casting back later
    original_dtype = df[group_col].dtype

    # ── 2. Expand counts → individual events (equivalent to tidyr::uncount) ─
    groups_expanded = np.repeat(df[group_col].values, df[count_col].values)
    n = len(groups_expanded)

    if n == 0:
        # Edge case: no events at all → return zeros
        out = pd.DataFrame({
            group_col: df[group_col].values,
            f"{new_col}_L": 0.0,
            f"{new_col}_U": 0.0,
        })
        return out

    # Progress message (matching R: "Running B bootstrap samples on n events…")
    if B > 100 and n > 1000:
        print(f"Running {B} bootstrap samples on {n} events for {count_col}...")

    # ── 3. Factor-encode groups ─────────────────────────────────────────────
    unique_groups, g_idx = np.unique(groups_expanded, return_inverse=True)
    G = len(unique_groups)

    # ── 4. Sparse one-hot matrix  (n × G) ──────────────────────────────────
    onehot = sparse.csr_matrix(
        (np.ones(n, dtype=np.float64), (np.arange(n), g_idx)),
        shape=(n, G),
    )

    # ── 5. Multinomial bootstrap: draw B samples ───────────────────────────
    #   R: W <- rmultinom(B, size = n, prob = rep(1/n, n))  → (n × B)
    #   numpy multinomial returns (B × n), so we transpose.
    prob = np.full(n, 1.0 / n)
    W = rng.multinomial(n, prob, size=B).T.astype(np.float64)  # (n, B)

    # ── 6. Aggregate: group_counts = onehot.T @ W  →  (G × B) ─────────────
    group_counts = (onehot.T @ W).toarray() if sparse.issparse(onehot.T @ W) else (onehot.T @ W)
    # onehot.T is (G, n); W is (n, B) → result is (G, B) dense

    # ── 7. Convert to percentages if requested ─────────────────────────────
    if use_percentages:
        col_sums = group_counts.sum(axis=0, keepdims=True)  # (1, B)
        col_sums[col_sums == 0] = 1  # guard against division by zero
        group_values = (group_counts / col_sums) * 100.0
    else:
        group_values = group_counts.astype(np.float64)

    # ── 8. Quantiles ───────────────────────────────────────────────────────
    alpha = 1.0 - conf_level
    lower_prob = alpha / 2.0
    upper_prob = 1.0 - alpha / 2.0

    lower_values = np.quantile(group_values, lower_prob, axis=1)  # (G,)
    upper_values = np.quantile(group_values, upper_prob, axis=1)  # (G,)

    # ── 9. Build result DataFrame ──────────────────────────────────────────
    out = pd.DataFrame({
        group_col: unique_groups,
        f"{new_col}_L": lower_values,
        f"{new_col}_U": upper_values,
    })

    # Cast group column back to original type (matches R behaviour)
    try:
        out[group_col] = out[group_col].astype(original_dtype)
    except (ValueError, TypeError):
        pass  # keep as-is if cast fails

    return out
