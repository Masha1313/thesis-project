# Molecular Similarity Experiment

Short file map for the project.

### `molecular_similarity_experiment.py`

Main implementation file. It contains:

- molecule filtering by heavy atom count;
- 3D conformer generation from SMILES;
- molecule voxelization;
- Zernike descriptor calculation;
- Zernike similarity calculation;
- RDKit USR distance calculation;
- RDKit Shape Tanimoto calculation;
- one-vs-all molecule comparison;
- top-k coverage analysis;
- multi-seed experiment runners;
- coverage and correlation summary builders;
- helper functions for pairwise correlation experiments.

### `run_fresh_coverage_tables.py`

Command-line script for running the final coverage experiment.

It imports the main functions from `molecular_similarity_experiment.py`, runs the experiment for selected seeds, and writes the result tables to CSV files.

### `z3dzernike.ipynb`

Notebook for viewing the final Zernike/RDKit coverage tables.

The long experiment run is commented out there because the final CSV outputs are already saved.

### `correlation_experiments.ipynb`

Notebook for exploratory pairwise correlation experiments and diagnostic plots.

### `chembl_raw_dump.csv`

Input ChEMBL dataset with:

- `chembl_id`;
- `smiles`.

### `fresh_seeds10_20_30_40_50_60_70_*.csv`

Final output files from the multi-seed experiment.

Most important outputs:

- `fresh_seeds10_20_30_40_50_60_70_table_all.csv`;
- `fresh_seeds10_20_30_40_50_60_70_table_at_least_one.csv`;
- `fresh_seeds10_20_30_40_50_60_70_correlation_summary.csv`.

### `pyproject.toml`

Project dependencies and Python package configuration.

### `uv.lock`

Locked dependency versions for reproducible runs with `uv`.
