# Understanding NYC Real Estate Market using Parallelization Techniques

Course project for CSYE7105 (HPC and AI). The goal is to measure end to end speedups from CPU parallelism and GPU acceleration across data preprocessing and model training workflows using the NYC property sales dataset.

## Repository layout

- `notebooks/`
  - `00_data_exploration.ipynb` data overview and EDA
  - `01_cpu_baselines.ipynb` serial CPU baseline plus CPU parallel variants and plots
  - `02_gpu_runner.ipynb` runs GPU scripts
- `scripts/` runnable scripts for CPU and GPU runs
- `data/` expected local datasets (ignored by git)
- `models/` saved model artifacts (ignored by git)
- `results/` timing logs and metrics (ignored by git)

## Dataset

This project uses the NYC Department of Finance rolling sales data. Download it locally and place it at:

- `data/raw/nyc-property-sales.csv`

Official sources:
- NYC DOF Rolling Sales data page (Excel and PDF by borough)
- NYC Open Data `Rolling Sales` dataset (API export available)

## Quickstart

Create an environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run notebooks:

```bash
jupyter lab
```

Or run scripts directly:

```bash
python scripts/pytorch_cpu_scaling_2_4.py
python scripts/pytorch_cpu_scaling_16_32_56.py
python scripts/pytorch_gpu.py
python scripts/xgboost_hpo_gpu.py
```

Note: `scripts/xgboost_gpu.py` is a placeholder because it was not included in the zip. Add your implementation or update the GPU notebook accordingly.

## Repro tips

- Notebooks are committed with outputs cleared to keep the repo small.
- Large datasets and artifacts are excluded via `.gitignore`.
- If you want to version large artifacts, use Git LFS or upload them as GitHub Releases instead of committing to the main repo.
