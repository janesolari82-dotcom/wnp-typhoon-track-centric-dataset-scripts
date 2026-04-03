# scripts

## Overview

This directory contains the scripts needed to reconstruct the released dataset, run the technical validation workflow, and reproduce the main project figures.

The scripts use project-relative path logic based on the repository layout.

## Dataset Build Scripts

### `01_data_processing/01_preprocess`

These scripts prepare the annual raster inputs used later in the typhoon-centered NetCDF workflow.

- `s01_nightlight_harmonization.py`
  - Harmonizes DMSP and VIIRS nightlight data into a consistent annual 2000-2024 series.
- `s02_population_log_pchip.py`
  - Reconstructs annual population rasters for 2000-2024 using the log-PCHIP workflow.
- `s03_gadm_gdp_preprocess.py`
  - Prepares GADM country geometry and World Bank GDP inputs for LitPop.
- `s04_litpop_2000_2024.py`
  - Builds annual LitPop rasters for 2000-2024.
- `config/*.yaml`
  - Configuration files used by the preprocessing scripts.

### `01_data_processing/02_track_base`

These scripts generate the yearly base typhoon NetCDF files.

- `s01_filter_ibtracs_wp.py`
  - Filters raw IBTrACS records to western North Pacific storms (`BASIN='WP'`) for 2000-2024.
- `s02_convert_ibtracs_nc.py`
  - Converts the filtered IBTrACS CSV into yearly base NetCDF files (`typhoon_{year}_base.nc`).

### `01_data_processing/03_legacy_provenance_ocean`

- `s01_legacy_integrate_copernicus_v3.py`
  - Legacy ocean integration script that adds Copernicus ocean variables to the base typhoon files and produces the old `ocean` intermediate stage.

### `01_data_processing/04_release_ocean_fix`

- `s01_reintegrate_copernicus_split_final.py`
  - Rebuilds ocean variables in the final yearly typhoon NetCDF files from split Copernicus inputs and writes the corrected release output.

### `01_data_processing/05_window_and_exposure_integration`

These scripts add exposure layers to the typhoon-centered NetCDF structure.

- `s01_integrate_litpop_windows.py`
  - Inserts annual LitPop windows into yearly typhoon NetCDF files.
- `s02_integrate_pop_light_windows.py`
  - Inserts annual population and nightlight windows into yearly typhoon NetCDF files.

### `01_data_processing/06_era5_emdat_integration`

These scripts integrate hazard variables and loss attribution.

- `s00_common.py`
  - Shared paths and helper functions for the ERA5 + EM-DAT stage.
- `s01_integrate_era5_daily_windows.py`
  - Integrates ERA5 daily wind and precipitation fields and derives hazard proxy variables.
- `s02_integrate_emdat_attribution.py`
  - Matches EM-DAT records, applies spatial fallback logic, allocates losses conservatively, and writes the audit tables.
- `s03_validate_emdat_workflow_v4.py`
  - Validates the ERA5 + EM-DAT integration outputs.
- `s04_run_emdat_workflow_v4.py`
  - Convenience runner for the ERA5 + EM-DAT workflow.

## Technical Validation Scripts

### `02_technical_validation/benchmark`

This folder contains the shared legacy validation modules still used by `benchmark_v2`.

- `s00_core_common.py`
  - Shared benchmark configuration, artifact writing, and result helpers.
- `core_common.py`
  - Compatibility alias retained for legacy loaders.
- `s01_l0_integrity.py`
  - Legacy L0 release-integrity implementation.
- `s02_l1_schema.py`
  - Legacy L1 schema and metadata validation implementation.
- `s03_l2_track_physics.py`
  - Legacy L2 track-realism implementation.
- `s04_l3_alignment.py`
  - Legacy L3 alignment and coverage implementation.
- `s05_l4_hazard_formula.py`
  - Legacy L4 hazard-formula implementation.
- `s06_l5_emdat.py`
  - Legacy L5 attribution-quality implementation.

### `02_technical_validation/benchmark/L6-7`

This folder preserves the legacy external-validation helpers that are still loaded by the current L6-L7 runner.

- `l67_common.py`
  - Shared helpers for the legacy L6-L7 validation code.
- `l6_external_validation.py`
  - Legacy external-validation implementation used as a compatibility dependency.
- `l7_case_validation.py`
  - Legacy case-validation implementation retained with the compatibility set.
- `run_l6_l7.py`
  - Old combined L6-L7 runner kept for provenance.
- `make_haiyan_gif_compare.py`
  - Legacy diagnostic plotting utility retained with the legacy set.

### `02_technical_validation/benchmark_v2`

This is the current technical-validation workflow.

- `s00_common_v2.py`
  - Shared helpers and layout logic for benchmark v2.
- `s01_l0_release_integrity_v2.py`
  - L0: release-package integrity checks.
- `s02_l1_schema_metadata_v2.py`
  - L1: schema and metadata consistency checks.
- `s03_l2_track_realism_v2.py`
  - L2: track realism and variable-range checks.
- `s04_l3_alignment_coverage_v2.py`
  - L3: window alignment and coverage checks.
- `s05_l4_hazard_sanity_v2.py`
  - L4: hazard-variable sanity and recomputation checks.
- `s06_l5_attribution_quality_v2.py`
  - L5: attribution quality and conservation checks.
- `s07_l6_external_gdis_imerg_v2.py`
  - L6: external validation against GDIS and IMERG.
- `s08_l7_imerg_case_validation_v2.py`
  - L7: event-level case validation built on the L6 context.
- `s09_run_core_l0_l5_v2.py`
  - Runner for the core validation stages L0-L5.
- `s10_run_l6_l7_v2.py`
  - Runner for the external validation stages L6-L7.
- `s11_run_all_v2.py`
  - One-shot runner for the full L0-L7 validation workflow.

### `02_technical_validation/figure_generation`

- `s01_generate_technical_validation_l0_l6_figure.py`
  - Generates the L0-L6 technical-validation overview figure.
- `s02_generate_external_validation_case_map_figure.py`
  - Generates the representative external-validation case map figure.

## Project Figure Script

### `03_project_visualization`

- `s01_make_multisource_data_display.py`
  - Generates the multi-panel overview figure of the core gridded data layers.

## Recommended Execution Order

The numbering is mainly for readability inside each folder. It is not a single, flat end-to-end sequence across the whole directory.

A practical order for rebuilding the data products is:

1. Run the preprocessing scripts in `01_data_processing/01_preprocess`.
2. Build the yearly base typhoon NetCDF files with the scripts in `01_data_processing/02_track_base`.
3. Run the legacy ocean integration in `01_data_processing/03_legacy_provenance_ocean`.
4. Add exposure windows with the scripts in `01_data_processing/05_window_and_exposure_integration`.
5. Run the ERA5 + EM-DAT stage in `01_data_processing/06_era5_emdat_integration`.
6. Apply the release ocean fix in `01_data_processing/04_release_ocean_fix` to generate the corrected release output in `new_final_output`.

For the validation workflow, use one of the following:

- `02_technical_validation/benchmark_v2/s09_run_core_l0_l5_v2.py` for L0-L5 only
- `02_technical_validation/benchmark_v2/s10_run_l6_l7_v2.py` for L6-L7 only
- `02_technical_validation/benchmark_v2/s11_run_all_v2.py` for the full L0-L7 workflow

After the required validation outputs exist, run the figure-generation scripts in:

- `02_technical_validation/figure_generation`
- `03_project_visualization`

## Legacy Ocean Integration Note

The legacy ocean integration script is kept because it is part of the provenance of the older workflow and of the older `04_final_output` files.

However, the released dataset in `reproduction_v6/new_final_output` is associated with the later correction step in:

- `01_data_processing/04_release_ocean_fix/s01_reintegrate_copernicus_split_final.py`

This distinction matters:

- `s01_legacy_integrate_copernicus_v3.py` explains how the old ocean stage was originally produced.
- `s01_reintegrate_copernicus_split_final.py` is the release-fix step associated with the current final output.

In other words, the legacy script is retained for provenance, while the reintegration script is the one tied to the released dataset.

## Running the Scripts

Most entry scripts support `--help`, for example:

```bash
python reproduction_v6/all_scripts/01_data_processing/01_preprocess/s01_nightlight_harmonization.py --help
python reproduction_v6/all_scripts/01_data_processing/06_era5_emdat_integration/s04_run_emdat_workflow_v4.py --help
python reproduction_v6/all_scripts/02_technical_validation/benchmark_v2/s11_run_all_v2.py --help
```

For the few scripts without a full argparse interface, a safe first check is to import them without entering `main()`.

## Notes

- This folder focuses on the scripts needed to rebuild, validate, and explain the released dataset.
- Data-download scripts are intentionally not included here.
- The legacy `benchmark/L6-7` folder is kept because the current L6-L7 startup path still depends on it.
