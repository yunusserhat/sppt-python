"""
Map creation for SPPT bivariate results.

Mirrors the base-R ``plot()`` map in sppt.R (lines 249-295) and the
ggplot2 publication-quality map in SPPT_Aggregated_Data.R (lines 495-548).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def create_bivariate_map(
    data,
    count_col: list[str],
    export_maps: bool = False,
    export_dir: str | None = None,
    map_dpi: int = 300,
) -> None:
    """Create a choropleth map of SIndex_Bivariate (-1 / 0 / 1).

    Parameters
    ----------
    data : GeoDataFrame
        Must have geometry and an ``SIndex_Bivariate`` column.
    count_col : list[str]
        [base_name, test_name] used in the legend labels.
    export_maps : bool
        Whether to save the map to disk.
    export_dir : str or None
        Directory for the exported PNG.  Defaults to cwd.
    map_dpi : int
        Resolution for exported map (default 300).
    """
    import geopandas as gpd  # deferred import to avoid hard dep at module level
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    print("Creating map...")

    try:
        base_name = count_col[0]
        test_name = count_col[1]

        # Colour scheme matching R:  gray80 → white → black  for -1, 0, 1
        cmap = ListedColormap(["#CCCCCC", "white", "black"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        data.plot(
            column="SIndex_Bivariate",
            cmap=cmap,
            norm=norm,
            edgecolor="#4D4D4D",  # gray30
            linewidth=0.3,
            ax=ax,
            legend=False,
        )

        # Custom legend (matching R's legend() call)
        legend_elements = [
            Patch(facecolor="#CCCCCC", edgecolor="black",
                  label=f"{base_name} > {test_name}"),
            Patch(facecolor="white", edgecolor="black",
                  label="Insignificant change"),
            Patch(facecolor="black", edgecolor="black",
                  label=f"{test_name} > {base_name}"),
        ]
        ax.legend(
            handles=legend_elements,
            title="Spatial Pattern",
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.9,
        )

        ax.set_title("S-Index Bivariate", fontsize=12, fontweight="bold")
        ax.set_axis_off()

        plt.tight_layout()

        if export_maps:
            if export_dir is None:
                export_dir = os.getcwd()
            else:
                os.makedirs(export_dir, exist_ok=True)
            filepath = os.path.join(export_dir, "map_bivariate_s_index.png")
            fig.savefig(filepath, dpi=map_dpi, bbox_inches="tight")
            print(f"Map exported successfully to: {export_dir}")
            print(f"  - map_bivariate_s_index.png\n")
        else:
            plt.show()
            print("Map created successfully.\n")

        plt.close(fig)

    except Exception as e:
        print(f"Note: Could not create/export map. Error: {e}\n")


def create_publication_map(
    data,
    count_col: list[str],
    export_path: str | None = None,
    map_dpi: int = 300,
) -> None:
    """Create a publication-quality choropleth (blue / white / red).

    Mirrors the ggplot2 map in SPPT_Aggregated_Data.R (lines 495-548).
    Uses matplotlib as the Python counterpart.

    Parameters
    ----------
    data : GeoDataFrame
        Must have geometry and an ``SIndex_Bivariate`` column.
    count_col : list[str]
        [base_name, test_name].
    export_path : str or None
        Full file path for export.  If None, displays interactively.
    map_dpi : int
        Resolution for export.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    try:
        base_name = count_col[0]
        test_name = count_col[1]

        # Publication colours: blue / white / red  for -1, 0, 1
        cmap = ListedColormap(["blue", "white", "red"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        data.plot(
            column="SIndex_Bivariate",
            cmap=cmap,
            norm=norm,
            edgecolor="#4D4D4D",  # gray30
            linewidth=0.3,
            ax=ax,
            legend=False,
        )

        legend_elements = [
            Patch(facecolor="blue", edgecolor="black", label="Base > Test"),
            Patch(facecolor="white", edgecolor="black",
                  label="Insignificant change"),
            Patch(facecolor="red", edgecolor="black", label="Test > Base"),
        ]
        ax.legend(
            handles=legend_elements,
            title="Spatial Pattern",
            loc="upper left",
            fontsize=10,
            title_fontsize=10,
            framealpha=0.0,
            edgecolor="none",
        )

        ax.set_title(
            "SPPT Bivariate Comparison",
            fontsize=14,
            fontweight="bold",
            ha="center",
        )
        ax.text(
            0.5, 1.01,
            f"Base: {base_name} vs Test: {test_name}",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
        )
        ax.set_axis_off()

        plt.tight_layout()

        if export_path:
            os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
            fig.savefig(export_path, dpi=map_dpi, bbox_inches="tight")
            print(f"Publication map exported to: {export_path}")
        else:
            plt.show()

        plt.close(fig)

    except Exception as e:
        print(f"Note: Could not create publication map. Error: {e}\n")
