"""
Vancouver crime SPPT example — Python reimplementation.

Replicates the R script SPPT_Aggregated_Data_RPackage_Code.R and the
publication-quality map from SPPT_Aggregated_Data.R.

Usage (from workspace root):
    python_sppt\\.venv\\Scripts\\python -m examples.run_vancouver

Or from python_sppt/:
    ../.venv/Scripts/python -m examples.run_vancouver
"""

from __future__ import annotations

import os
import sys

# Ensure the package is importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless runs

from sppt import sppt, create_publication_map


def main():
    # ── 1. Locate the shapefile bundled with the R package ─────────────────
    # The shapefile lives at inst/extdata/Vancouver_DAs_Crime_2021.shp
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    shp_path = os.path.join(
        workspace_root, "inst", "extdata", "Vancouver_DAs_Crime_2021.shp"
    )

    if not os.path.exists(shp_path):
        print(f"ERROR: Shapefile not found at {shp_path}")
        sys.exit(1)

    print(f"Loading shapefile: {shp_path}")
    data = gpd.read_file(shp_path)

    # ── 2. Transform CRS to EPSG:26910 (NAD83 / UTM Zone 10N) ─────────────
    data = data.to_crs(epsg=26910)

    print(f"Data loaded: {len(data)} rows, columns: {list(data.columns)}")
    print(f"CRS: {data.crs}")
    print()

    # ── 3. Run SPPT (mirrors the R example exactly) ───────────────────────
    result = sppt(
        data=data,
        group_col="DAUID",               # unique identifier for spatial groups
        count_col=["TFV", "TOV"],         # base variable, test variable
        B=200,
        conf_level=0.95,                  # 95% confidence interval
        check_overlap=True,
        create_maps=True,
        export_maps=False,                # set True to save map to disk
        export_dir=None,
        map_dpi=300,
        export_results=False,             # set True to export results
        export_format="shp",
        export_results_dir=None,
        seed=171717,                      # for reproducibility
        use_percentages=True,             # test changes in spatial distribution
        fix_base=False,                   # bootstrap both variables
    )

    # ── 4. Print S-Index results (matching R: attr(result, "s_index")) ────
    print(f"S-Index:        {result.s_index}")
    print(f"Robust S-Index: {result.robust_s_index}")
    print()

    # ── 5. Show first few rows of the result ──────────────────────────────
    cols_of_interest = [
        "DAUID", "TFV", "TOV",
        "TFV_L", "TFV_U", "TOV_L", "TOV_U",
        "intervals_overlap", "SIndex_Bivariate",
    ]
    existing = [c for c in cols_of_interest if c in result.data.columns]
    print("Sample output:")
    print(result.data[existing].head(10).to_string(index=False))
    print()

    # ── 6. Publication-quality map (blue / white / red) ───────────────────
    print("Creating publication-quality map...")
    pub_map_path = os.path.join(script_dir, "SIndex_Bivariate_Map.png")
    create_publication_map(
        result.data,
        count_col=["TFV", "TOV"],
        export_path=pub_map_path,
        map_dpi=300,
    )

    print("Done!")


if __name__ == "__main__":
    main()
