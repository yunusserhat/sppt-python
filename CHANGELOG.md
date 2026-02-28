# Changelog

## 0.1.0 (2026-02-28)

### Initial Release

- Core `sppt()` function with full parameter support
- Bootstrap resampling engine using sparse matrices (`scipy.sparse`)
- S-Index and Robust S-Index calculation
- SIndex_Bivariate per-unit directional change indicator
- Percentage mode (spatial distribution) and count mode
- Fixed base variable option
- Choropleth map generation (standard + publication quality)
- Export to Shapefile, GeoPackage, CSV, TXT, Pickle
- Bundled Vancouver DA Crime 2021 sample dataset (1,019 polygons)
- `load_sample_data()` convenience function
- Google Colab-compatible Jupyter notebooks
- 28 unit tests
- Based on the R package [`sppt.aggregated.data`](https://github.com/martin-a-andresen/sppt.aggregated.data) by Martin A. Andresen
