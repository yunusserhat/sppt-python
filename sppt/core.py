"""
Main ``sppt()`` function – the public API.

Orchestrates bootstrap, overlap, mapping, and export, faithfully mirroring the
R function in ``sppt.R`` (lines 78-380).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_single_variable
from .overlap import compute_overlap, compute_s_indices, compute_sindex_bivariate
from .mapping import create_bivariate_map
from .export import export_results as _export_results


# ═══════════════════════════════════════════════════════════════════════════════
# Result container  (Python equivalent of R's attr() on the returned data)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SPPTResult:
    """Container returned by :func:`sppt`.

    Attributes
    ----------
    data : DataFrame or GeoDataFrame
        The input data augmented with CI bounds, overlap, and SIndex columns.
    s_index : float or None
        Global S-Index (set when ``check_overlap=True``).
    robust_s_index : float or None
        Robust S-Index excluding all-zero rows.
    fix_base : bool
        Whether the base variable was fixed.
    use_percentages : bool
        Whether percentage mode was used.
    """

    data: Any  # DataFrame | GeoDataFrame
    s_index: float | None = None
    robust_s_index: float | None = None
    fix_base: bool = False
    use_percentages: bool = True

    # Convenience accessors ──────────────────────────────────────────────────
    def __getitem__(self, key):
        """Allow ``result["col"]`` to access the underlying DataFrame."""
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        lines = [f"SPPTResult(rows={len(self.data)})"]
        if self.s_index is not None:
            lines.append(f"  S-Index:        {self.s_index:.4f}")
            lines.append(f"  Robust S-Index: {self.robust_s_index:.4f}")
        lines.append(f"  fix_base={self.fix_base}, use_percentages={self.use_percentages}")
        lines.append(f"  Columns: {list(self.data.columns)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main function
# ═══════════════════════════════════════════════════════════════════════════════

def sppt(
    data,
    group_col: str = "group",
    count_col: str | list[str] | None = None,
    B: int = 200,
    new_col: str | list[str] | None = None,
    seed: int | None = None,
    conf_level: float = 0.95,
    check_overlap: bool = False,
    fix_base: bool = False,
    use_percentages: bool = True,
    create_maps: bool = True,
    export_maps: bool = False,
    export_dir: str | None = None,
    map_dpi: int = 300,
    export_results: bool = False,
    export_format: str = "shp",
    export_results_dir: str | None = None,
) -> SPPTResult:
    """Spatial Point Pattern Test for aggregated data.

    Performs bootstrap-based spatial pattern point tests on aggregated count
    data.  Compares spatial distributions between variables and calculates
    S-Index metrics to quantify spatial pattern overlap.

    Parameters
    ----------
    data : DataFrame or GeoDataFrame
        Aggregated count data.  If a GeoDataFrame, geometry is preserved.
    group_col : str
        Column identifying spatial groups (default ``"group"``).
    count_col : str or list[str]
        Column(s) with integer count data.  For bivariate analysis provide
        two names: ``[base, test]``.
    B : int
        Number of bootstrap samples (default 200).
    new_col : str, list[str], or None
        Output column-name prefix(es).  Defaults to *count_col* names.
    seed : int or None
        Random seed for reproducibility.
    conf_level : float
        Confidence level, e.g. 0.95 for 95 % CI (default).
    check_overlap : bool
        Calculate overlap and S-Index statistics (default ``False``).
    fix_base : bool
        Fix the first (base) variable without bootstrapping (default ``False``).
    use_percentages : bool
        Convert to percentages (True) or keep counts (False).
    create_maps : bool
        Generate choropleth map for bivariate case (default ``True``).
    export_maps : bool
        Save the map to disk (default ``False``).
    export_dir : str or None
        Directory for exported map.
    map_dpi : int
        DPI for exported map (default 300).
    export_results : bool
        Save the result data to disk (default ``False``).
    export_format : str
        File format: ``"shp"``, ``"csv"``, ``"txt"``, ``"gpkg"``, ``"pickle"``
        (default ``"shp"``).
    export_results_dir : str or None
        Directory for results export.

    Returns
    -------
    SPPTResult
        A container holding the augmented DataFrame/GeoDataFrame along with
        S-Index metadata.  Access the data via ``result.data``.
    """
    import geopandas as gpd

    # ── Normalise count_col / new_col to lists ─────────────────────────────
    if count_col is None:
        raise ValueError("count_col must be specified")
    if isinstance(count_col, str):
        count_col = [count_col]
    if new_col is None:
        new_col = list(count_col)
    elif isinstance(new_col, str):
        new_col = [new_col]
    if len(new_col) != len(count_col):
        raise ValueError("new_col must have the same length as count_col")

    # ── Multi-variable processing ──────────────────────────────────────────
    result = data.copy()
    is_geo = isinstance(result, gpd.GeoDataFrame)

    for i, (ccol, ncol) in enumerate(zip(count_col, new_col)):
        if fix_base and i == 0:
            # Fix base: set _L = _U = exact percentage (or count)
            if use_percentages:
                total_count = result[ccol].sum()
                if total_count > 0:
                    pct = (result[ccol] / total_count) * 100.0
                else:
                    pct = 0.0
                result[f"{ncol}_L"] = pct
                result[f"{ncol}_U"] = pct
            else:
                result[f"{ncol}_L"] = result[ccol].astype(float)
                result[f"{ncol}_U"] = result[ccol].astype(float)
        else:
            # Bootstrap this variable
            var_seed = (seed + i) if seed is not None else None  # R uses seed + i - 1 (1-based)
            ci_df = bootstrap_single_variable(
                data=result,
                group_col=group_col,
                count_col=ccol,
                B=B,
                new_col=ncol,
                seed=var_seed,
                conf_level=conf_level,
                use_percentages=use_percentages,
            )

            # Merge CI bounds back (left join on group_col)
            l_col = f"{ncol}_L"
            u_col = f"{ncol}_U"
            # Drop existing columns if they exist (from a previous iteration)
            for c in [l_col, u_col]:
                if c in result.columns:
                    result = result.drop(columns=c)

            result = result.merge(ci_df, on=group_col, how="left")

            # Replace NaN with 0 (matches R's replace_na)
            result[l_col] = result[l_col].fillna(0.0)
            result[u_col] = result[u_col].fillna(0.0)

    # ── Overlap & S-Index ──────────────────────────────────────────────────
    stats_dict: dict | None = None

    if check_overlap and len(count_col) > 1:
        result = compute_overlap(result, count_col, new_col, fix_base=fix_base)

        if len(count_col) == 2:
            result = compute_sindex_bivariate(result, count_col)

        stats_dict = compute_s_indices(
            result, count_col, fix_base=fix_base, use_percentages=use_percentages
        )

        # ── Map ────────────────────────────────────────────────────────────
        if create_maps and is_geo and len(count_col) == 2:
            create_bivariate_map(
                result,
                count_col,
                export_maps=export_maps,
                export_dir=export_dir,
                map_dpi=map_dpi,
            )

    # ── Export ─────────────────────────────────────────────────────────────
    if export_results and len(count_col) > 1:
        _export_results(
            result,
            count_col,
            export_format=export_format,
            export_results_dir=export_results_dir,
        )

    # ── Build return value ─────────────────────────────────────────────────
    return SPPTResult(
        data=result,
        s_index=stats_dict["s_index"] if stats_dict else None,
        robust_s_index=stats_dict["robust_s_index"] if stats_dict else None,
        fix_base=fix_base,
        use_percentages=use_percentages,
    )
