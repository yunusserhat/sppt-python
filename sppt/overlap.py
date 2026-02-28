"""
Overlap detection, S-Index, Robust S-Index, and SIndex_Bivariate.

Faithfully mirrors the R implementation in sppt.R (lines 130-225).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_overlap(
    data: pd.DataFrame,
    count_col: list[str],
    new_col: list[str],
    fix_base: bool = False,
) -> pd.DataFrame:
    """Compute the ``intervals_overlap`` column (0/1) for each row.

    Parameters
    ----------
    data : DataFrame or GeoDataFrame
        Must already contain ``{new_col[i]}_L`` and ``{new_col[i]}_U`` columns.
    count_col : list[str]
        Original count column names (length >= 2).
    new_col : list[str]
        Column-name prefixes used for the CI bounds.
    fix_base : bool
        If True, the base variable has a point estimate (L == U) and
        the overlap check becomes "does the base value fall within the
        test interval?".

    Returns
    -------
    DataFrame
        Same data with an ``intervals_overlap`` column appended.
    """
    result = data.copy()

    if fix_base:
        # When fix_base is TRUE, check if Base value falls within Test interval
        # base_L == base_U (point estimate), so this is:
        #   base_value >= test_L  AND  base_value <= test_U
        base_lower = f"{new_col[0]}_L"
        base_upper = f"{new_col[0]}_U"
        test_lower = f"{new_col[1]}_L"
        test_upper = f"{new_col[1]}_U"

        result["intervals_overlap"] = (
            (result[base_lower] >= result[test_lower])
            & (result[base_upper] <= result[test_upper])
        ).astype(int)
    else:
        # General case: intervals overlap if max(all lowers) <= min(all uppers)
        lower_cols = [f"{nc}_L" for nc in new_col]
        upper_cols = [f"{nc}_U" for nc in new_col]

        lowers = result[lower_cols].values  # (N, k)
        uppers = result[upper_cols].values  # (N, k)

        max_lower = np.nanmax(lowers, axis=1)
        min_upper = np.nanmin(uppers, axis=1)

        result["intervals_overlap"] = (max_lower <= min_upper).astype(int)

    return result


def compute_sindex_bivariate(
    data: pd.DataFrame,
    count_col: list[str],
) -> pd.DataFrame:
    """Add ``SIndex_Bivariate`` column for exactly two variables.

    Values:
        0  → intervals overlap (no significant difference)
        1  → test > base  (raw count)
       -1  → base > test  (raw count)

    Mirrors R's ``case_when`` logic in sppt.R lines 176-186.
    """
    result = data.copy()
    base_var = count_col[0]
    test_var = count_col[1]

    sindex = np.where(
        result["intervals_overlap"] == 1,
        0,
        np.where(
            result[test_var] > result[base_var],
            1,
            np.where(
                result[test_var] < result[base_var],
                -1,
                0,  # tie
            ),
        ),
    )
    result["SIndex_Bivariate"] = sindex.astype(int)
    return result


def compute_s_indices(
    data: pd.DataFrame,
    count_col: list[str],
    fix_base: bool = False,
    use_percentages: bool = True,
) -> dict:
    """Calculate S-Index and Robust S-Index and print formatted statistics.

    Parameters
    ----------
    data : DataFrame or GeoDataFrame
        Must contain ``intervals_overlap`` column and the original count columns.
    count_col : list[str]
        Original count column names.
    fix_base : bool
        Whether the base variable was fixed.
    use_percentages : bool
        Whether percentages mode was used.

    Returns
    -------
    dict
        Keys: ``s_index``, ``robust_s_index``, ``fix_base``, ``use_percentages``.
    """
    total_obs = len(data)
    sum_overlap = int(data["intervals_overlap"].sum())

    # S-Index: proportion of all observations with overlapping intervals
    s_index = sum_overlap / total_obs if total_obs > 0 else 0.0

    # Robust S-Index: exclude rows where ALL count columns are zero
    # Need to drop geometry if present
    try:
        count_data = data[count_col].copy()
    except KeyError:
        count_data = data.drop(columns="geometry", errors="ignore")[count_col].copy()

    nonzero_mask = (count_data > 0).any(axis=1).values
    nonzero_obs = int(nonzero_mask.sum())
    sum_overlap_nonzero = int(data.loc[nonzero_mask, "intervals_overlap"].sum())

    robust_s_index = (
        sum_overlap_nonzero / nonzero_obs if nonzero_obs > 0 else float("nan")
    )

    # ── Print formatted statistics (matching R output exactly) ─────────────
    print()
    print("========================================")
    print("Spatial Pattern Overlap Statistics")
    if fix_base:
        print("Mode: Fixed Base (Test randomized)")
    if use_percentages:
        print("Using: Percentages (spatial distribution)")
    else:
        print("Using: Counts (absolute values)")
    print("========================================")
    print(f"S-Index:           {s_index:.4f}")
    print(f"Robust S-Index:    {robust_s_index:.4f}")
    print("----------------------------------------")
    print(f"Total observations:                 {total_obs}")
    print(f"Observations with overlap:          {sum_overlap}")
    print(f"Observations with non-zero counts:  {nonzero_obs}")
    print("========================================")
    print()

    return {
        "s_index": s_index,
        "robust_s_index": robust_s_index,
        "fix_base": fix_base,
        "use_percentages": use_percentages,
    }
