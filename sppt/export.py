"""
Result export for SPPT.

Mirrors the export logic in sppt.R (lines 305-370).
Supported formats: shp, csv, txt, gpkg, pickle (Python equivalent of R's rds).
"""

from __future__ import annotations

import os
import warnings


def export_results(
    data,
    count_col: list[str],
    export_format: str = "shp",
    export_results_dir: str | None = None,
) -> None:
    """Export SPPT results to disk.

    Parameters
    ----------
    data : DataFrame or GeoDataFrame
        The result data to export.
    count_col : list[str]
        Variable names used in the analysis (for building the filename).
    export_format : str
        One of ``"shp"``, ``"csv"``, ``"txt"``, ``"gpkg"``, ``"pickle"``.
    export_results_dir : str or None
        Directory for the output file.  Defaults to cwd.
    """
    import pandas as pd

    # Determine export directory
    if export_results_dir is None:
        export_results_dir = os.getcwd()
        print(f"Exporting results to current working directory: {export_results_dir}")
    else:
        os.makedirs(export_results_dir, exist_ok=True)

    # Build filename from variable names (matches R: sppt_output_{var1}_{var2}.ext)
    var_names = "_".join(count_col)
    base_filename = f"sppt_output_{var_names}"

    fmt = export_format.lower()

    try:
        if fmt == "shp":
            _export_spatial(data, export_results_dir, base_filename, "ESRI Shapefile", ".shp")
        elif fmt in ("csv", "txt"):
            _export_tabular(data, export_results_dir, base_filename, fmt)
        elif fmt == "gpkg":
            _export_spatial(data, export_results_dir, base_filename, "GPKG", ".gpkg")
        elif fmt == "pickle":
            filepath = os.path.join(export_results_dir, f"{base_filename}.pkl")
            pd.to_pickle(data, filepath)
            print(f"Results exported as pickle: {filepath}")
        else:
            warnings.warn(
                f"Unsupported export format: {export_format}. "
                "Supported formats: shp, csv, txt, gpkg, pickle"
            )
    except Exception as e:
        warnings.warn(f"Failed to export results: {e}")


# ── helpers ────────────────────────────────────────────────────────────────────

def _export_spatial(data, directory: str, base_filename: str, driver: str, ext: str):
    """Export as a spatial format (shapefile / geopackage)."""
    import geopandas as gpd

    if not isinstance(data, gpd.GeoDataFrame):
        warnings.warn(f"Cannot export as {driver}: data is not a GeoDataFrame")
        return

    filepath = os.path.join(directory, f"{base_filename}{ext}")
    # Remove existing file to mimic R's delete_dsn = TRUE
    if os.path.exists(filepath):
        os.remove(filepath)
    data.to_file(filepath, driver=driver)
    print(f"Results exported as {driver}: {filepath}")


def _export_tabular(data, directory: str, base_filename: str, fmt: str):
    """Export as CSV or TXT (geometry dropped)."""
    import geopandas as gpd
    import pandas as pd

    if isinstance(data, gpd.GeoDataFrame):
        df = pd.DataFrame(data.drop(columns="geometry"))
    else:
        df = data

    filepath = os.path.join(directory, f"{base_filename}.{fmt}")
    df.to_csv(filepath, index=False)
    print(f"Results exported as {fmt.upper()}: {filepath}")
