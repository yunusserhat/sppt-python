"""
Unit tests for the SPPT Python reimplementation.

Covers:
  - Single-variable bootstrap (CI bounds are reasonable)
  - Multi-variable bootstrap
  - Overlap detection (both fix_base modes)
  - S-Index / Robust S-Index calculation
  - SIndex_Bivariate logic
  - Edge cases (zero counts, single group)
  - Percentage vs count mode
  - Vancouver shapefile smoke test
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sppt import sppt, SPPTResult
from sppt.bootstrap import bootstrap_single_variable
from sppt.overlap import compute_overlap, compute_s_indices, compute_sindex_bivariate


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_simple_df():
    """Three groups with known counts."""
    return pd.DataFrame({
        "id": ["A", "B", "C"],
        "var1": [100, 50, 50],
        "var2": [50, 50, 100],
    })


def _make_zero_df():
    """All-zero counts."""
    return pd.DataFrame({
        "id": ["A", "B", "C"],
        "var1": [0, 0, 0],
        "var2": [0, 0, 0],
    })


def _make_numeric_group_df():
    """Groups identified by numeric IDs."""
    return pd.DataFrame({
        "gid": [1, 2, 3, 4],
        "base": [80, 10, 5, 5],
        "test": [5, 5, 10, 80],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBootstrapSingleVariable:

    def test_basic_ci_bounds(self):
        df = _make_simple_df()
        ci = bootstrap_single_variable(
            df, group_col="id", count_col="var1", B=500, seed=42,
            use_percentages=True,
        )
        assert f"var1_L" in ci.columns
        assert f"var1_U" in ci.columns
        assert len(ci) == 3  # three groups
        # Lower should be <= upper for every group
        assert (ci["var1_L"] <= ci["var1_U"]).all()

    def test_percentage_mode_sums_roughly_100(self):
        df = _make_simple_df()
        ci = bootstrap_single_variable(
            df, group_col="id", count_col="var1", B=500, seed=42,
            use_percentages=True,
        )
        # Mid-points should sum to ~100 within CI bounds
        midpoints = (ci["var1_L"] + ci["var1_U"]) / 2
        assert 90 < midpoints.sum() < 110  # loose check

    def test_count_mode(self):
        df = _make_simple_df()
        ci = bootstrap_single_variable(
            df, group_col="id", count_col="var1", B=500, seed=42,
            use_percentages=False,
        )
        # In count mode the bounds should be near the original counts
        # Group A has count 100 out of 200 total
        row_a = ci[ci["id"] == "A"].iloc[0]
        assert 80 < row_a["var1_U"] < 130

    def test_zero_counts(self):
        df = _make_zero_df()
        ci = bootstrap_single_variable(
            df, group_col="id", count_col="var1", B=100, seed=42,
        )
        assert (ci["var1_L"] == 0).all()
        assert (ci["var1_U"] == 0).all()

    def test_seed_reproducibility(self):
        df = _make_simple_df()
        ci1 = bootstrap_single_variable(df, "id", "var1", B=100, seed=999)
        ci2 = bootstrap_single_variable(df, "id", "var1", B=100, seed=999)
        pd.testing.assert_frame_equal(ci1, ci2)

    def test_numeric_group_column(self):
        df = _make_numeric_group_df()
        ci = bootstrap_single_variable(
            df, group_col="gid", count_col="base", B=100, seed=42,
        )
        # Group dtype should be preserved as numeric
        assert np.issubdtype(ci["gid"].dtype, np.integer) or np.issubdtype(ci["gid"].dtype, np.floating)

    def test_custom_new_col(self):
        df = _make_simple_df()
        ci = bootstrap_single_variable(
            df, "id", "var1", B=50, seed=1, new_col="myvar",
        )
        assert "myvar_L" in ci.columns
        assert "myvar_U" in ci.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Overlap tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlap:

    def test_overlapping_intervals(self):
        """Intervals that overlap → overlap = 1."""
        df = pd.DataFrame({
            "var1_L": [10.0], "var1_U": [30.0],
            "var2_L": [20.0], "var2_U": [40.0],
            "var1": [5], "var2": [5],
        })
        result = compute_overlap(df, ["var1", "var2"], ["var1", "var2"], fix_base=False)
        assert result["intervals_overlap"].iloc[0] == 1

    def test_non_overlapping_intervals(self):
        """Intervals that don't overlap → overlap = 0."""
        df = pd.DataFrame({
            "var1_L": [10.0], "var1_U": [20.0],
            "var2_L": [30.0], "var2_U": [40.0],
            "var1": [5], "var2": [5],
        })
        result = compute_overlap(df, ["var1", "var2"], ["var1", "var2"], fix_base=False)
        assert result["intervals_overlap"].iloc[0] == 0

    def test_fix_base_overlap(self):
        """fix_base: base point value within test interval → overlap = 1."""
        df = pd.DataFrame({
            "var1_L": [25.0], "var1_U": [25.0],  # point estimate
            "var2_L": [20.0], "var2_U": [30.0],   # interval containing 25
            "var1": [5], "var2": [5],
        })
        result = compute_overlap(df, ["var1", "var2"], ["var1", "var2"], fix_base=True)
        assert result["intervals_overlap"].iloc[0] == 1

    def test_fix_base_no_overlap(self):
        """fix_base: base value outside test interval → overlap = 0."""
        df = pd.DataFrame({
            "var1_L": [50.0], "var1_U": [50.0],
            "var2_L": [20.0], "var2_U": [30.0],
            "var1": [5], "var2": [5],
        })
        result = compute_overlap(df, ["var1", "var2"], ["var1", "var2"], fix_base=True)
        assert result["intervals_overlap"].iloc[0] == 0


class TestSIndexBivariate:

    def test_overlap_gives_zero(self):
        df = pd.DataFrame({
            "intervals_overlap": [1],
            "base": [10], "test": [20],
        })
        result = compute_sindex_bivariate(df, ["base", "test"])
        assert result["SIndex_Bivariate"].iloc[0] == 0

    def test_test_greater_gives_positive(self):
        df = pd.DataFrame({
            "intervals_overlap": [0],
            "base": [10], "test": [20],
        })
        result = compute_sindex_bivariate(df, ["base", "test"])
        assert result["SIndex_Bivariate"].iloc[0] == 1

    def test_base_greater_gives_negative(self):
        df = pd.DataFrame({
            "intervals_overlap": [0],
            "base": [20], "test": [10],
        })
        result = compute_sindex_bivariate(df, ["base", "test"])
        assert result["SIndex_Bivariate"].iloc[0] == -1

    def test_tie_no_overlap_gives_zero(self):
        df = pd.DataFrame({
            "intervals_overlap": [0],
            "base": [10], "test": [10],
        })
        result = compute_sindex_bivariate(df, ["base", "test"])
        assert result["SIndex_Bivariate"].iloc[0] == 0


class TestSIndices:

    def test_perfect_overlap(self):
        df = pd.DataFrame({
            "intervals_overlap": [1, 1, 1],
            "var1": [10, 20, 30], "var2": [10, 20, 30],
        })
        stats = compute_s_indices(df, ["var1", "var2"])
        assert stats["s_index"] == 1.0
        assert stats["robust_s_index"] == 1.0

    def test_no_overlap(self):
        df = pd.DataFrame({
            "intervals_overlap": [0, 0, 0],
            "var1": [10, 20, 30], "var2": [10, 20, 30],
        })
        stats = compute_s_indices(df, ["var1", "var2"])
        assert stats["s_index"] == 0.0
        assert stats["robust_s_index"] == 0.0

    def test_robust_excludes_zeros(self):
        """Rows where all count cols are 0 are excluded from robust S-Index."""
        df = pd.DataFrame({
            "intervals_overlap": [1, 1, 0],  # third row: both zero, no overlap
            "var1": [10, 0, 0],
            "var2": [0, 5, 0],
        })
        stats = compute_s_indices(df, ["var1", "var2"])
        # S-Index = 2/3
        assert abs(stats["s_index"] - 2 / 3) < 1e-10
        # Robust: nonzero rows are rows 0 and 1 (A has var1>0, B has var2>0)
        # Row 2 is all-zero → excluded
        # Among nonzero rows: both overlap → robust = 2/2 = 1.0
        assert stats["robust_s_index"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: full sppt() call
# ═══════════════════════════════════════════════════════════════════════════════

class TestSPPTIntegration:

    def test_basic_two_variable(self):
        df = _make_simple_df()
        result = sppt(
            df, group_col="id", count_col=["var1", "var2"],
            B=100, seed=42, check_overlap=True,
            create_maps=False, use_percentages=True,
        )
        assert isinstance(result, SPPTResult)
        assert "var1_L" in result.data.columns
        assert "var2_U" in result.data.columns
        assert "intervals_overlap" in result.data.columns
        assert "SIndex_Bivariate" in result.data.columns
        assert 0 <= result.s_index <= 1
        assert 0 <= result.robust_s_index <= 1

    def test_fix_base(self):
        df = _make_simple_df()
        result = sppt(
            df, group_col="id", count_col=["var1", "var2"],
            B=100, seed=42, check_overlap=True,
            fix_base=True, create_maps=False,
        )
        # Base variable _L == _U (point estimate)
        assert (result.data["var1_L"] == result.data["var1_U"]).all()
        assert result.fix_base is True

    def test_count_mode(self):
        df = _make_simple_df()
        result = sppt(
            df, group_col="id", count_col=["var1", "var2"],
            B=100, seed=42, check_overlap=True,
            use_percentages=False, create_maps=False,
        )
        assert result.use_percentages is False

    def test_single_variable(self):
        df = _make_simple_df()
        result = sppt(
            df, group_col="id", count_col="var1",
            B=50, seed=42, create_maps=False,
        )
        assert "var1_L" in result.data.columns
        assert "var1_U" in result.data.columns
        # No overlap columns for single variable
        assert "intervals_overlap" not in result.data.columns

    def test_numeric_groups(self):
        df = _make_numeric_group_df()
        result = sppt(
            df, group_col="gid", count_col=["base", "test"],
            B=100, seed=42, check_overlap=True, create_maps=False,
        )
        assert len(result.data) == 4
        assert result.s_index is not None

    def test_export_csv(self):
        df = _make_simple_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = sppt(
                df, group_col="id", count_col=["var1", "var2"],
                B=50, seed=42, check_overlap=True, create_maps=False,
                export_results=True, export_format="csv",
                export_results_dir=tmpdir,
            )
            expected = os.path.join(tmpdir, "sppt_output_var1_var2.csv")
            assert os.path.exists(expected)

    def test_export_pickle(self):
        df = _make_simple_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = sppt(
                df, group_col="id", count_col=["var1", "var2"],
                B=50, seed=42, check_overlap=True, create_maps=False,
                export_results=True, export_format="pickle",
                export_results_dir=tmpdir,
            )
            expected = os.path.join(tmpdir, "sppt_output_var1_var2.pkl")
            assert os.path.exists(expected)

    def test_result_repr(self):
        df = _make_simple_df()
        result = sppt(
            df, group_col="id", count_col=["var1", "var2"],
            B=50, seed=42, check_overlap=True, create_maps=False,
        )
        r = repr(result)
        assert "SPPTResult" in r
        assert "S-Index" in r


# ═══════════════════════════════════════════════════════════════════════════════
# Vancouver shapefile smoke test
# ═══════════════════════════════════════════════════════════════════════════════

class TestVancouverSmokeTest:

    @pytest.fixture
    def shp_path(self):
        p = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "inst", "extdata", "Vancouver_DAs_Crime_2021.shp",
        )
        p = os.path.abspath(p)
        if not os.path.exists(p):
            pytest.skip(f"Vancouver shapefile not found at {p}")
        return p

    def test_load_and_run(self, shp_path):
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")

        data = gpd.read_file(shp_path)
        data = data.to_crs(epsg=26910)

        result = sppt(
            data=data,
            group_col="DAUID",
            count_col=["TFV", "TOV"],
            B=100,  # fewer for speed in CI
            conf_level=0.95,
            check_overlap=True,
            create_maps=False,
            seed=171717,
            use_percentages=True,
            fix_base=False,
        )

        # Basic assertions
        assert isinstance(result, SPPTResult)
        assert 0 <= result.s_index <= 1
        assert 0 <= result.robust_s_index <= 1
        assert "TFV_L" in result.data.columns
        assert "TOV_U" in result.data.columns
        assert "intervals_overlap" in result.data.columns
        assert "SIndex_Bivariate" in result.data.columns
        assert set(result.data["SIndex_Bivariate"].unique()).issubset({-1, 0, 1})

    def test_fix_base_mode(self, shp_path):
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")

        data = gpd.read_file(shp_path)
        data = data.to_crs(epsg=26910)

        result = sppt(
            data=data,
            group_col="DAUID",
            count_col=["TFV", "TOV"],
            B=100,
            check_overlap=True,
            create_maps=False,
            seed=171717,
            fix_base=True,
        )

        # Base _L == _U (point estimate, no bootstrap)
        assert (result.data["TFV_L"] == result.data["TFV_U"]).all()
        assert result.fix_base is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
