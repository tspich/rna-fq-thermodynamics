# Inosine duplex melting — data and analysis

Raw melting data and analysis code to reproduce the main result of the paper:
the **Systematic measurement of thermodynamic nearest-neighbor parameters for
Inosine-containing double-stranded RNAs using a fluorophore-quencher-based
approach**, obtained by least-squares fitting of modified-vs-unmodified duplex
stability differences.

## Layout

```
.
├── environment.yml            conda environment
├── data/
│   ├── fluo_raw.csv.gz        raw fluorescence melting curves  (main result)
│   └── uv_raw.csv.gz          raw UV melting curves            (secondary)
├── melting/                   analysis package
│   ├── load_data.py           read curves -> fit -> Results
│   ├── util.py                per-curve melting fits
│   ├── methods.py, functions.py   fitting primitives
│   ├── results_class.py       Strand_res / Results (+ ViennaRNA model values)
│   ├── nn_fit.py              nearest-neighbour least squares (main result)
│   ├── variables.py, sequences.py, constants.py
│   └── params/                ViennaRNA modified-base parameter files
├── scripts/
│   ├── run_main_result.py     main result: per-curve fit -> NN ΔG/ΔH params
│   ├── run_global_fit.py      global (all-replicates) fit variant of the NN ΔG
│   └── run_uv.py              independent UV melting/annealing Tm and ΔG
└── results/                   output tables (written by the scripts)
```

## Setup

```bash
conda env create -f environment.yml
conda activate inosine-fq
```

## Reproduce the main result

```bash
python scripts/run_main_result.py
```

This fits every melting curve in `data/fluo_raw.csv.gz`, computes the
modified−unmodified stability differences, solves the nearest-neighbour least
squares, prints the full reports, and writes:

- `results/nn_params_dG.csv` — inosine NN ΔG parameters (kcal/mol) + std. error
- `results/nn_params_dH.csv` — companion ΔH parameters


## Other analyses

```bash
python scripts/run_global_fit.py   # global fit: all replicates of a duplex at once
python scripts/run_uv.py           # UV melting/annealing Tm + ΔG (independent check)
```

- `run_global_fit.py` fits all replicate curves of each duplex simultaneously
  (shared ΔH/ΔS, per-curve baselines) and runs the same NN least squares →
  `results/nn_params_dG_global.csv`.
- `run_uv.py` fits each UV scan (per wavelength and melting/annealing direction)
  → `results/uv_results.csv`.

## Data format

`fluo_raw.csv.gz` is long/tidy — one row per measured point:

| column | meaning |
|---|---|
| `strand` | duplex id, e.g. `26-31` |
| `oligo_c` | oligo concentration [µM] |
| `salt_c` | NaCl concentration [mM] |
| `replicate` | 0-based replicate index within a `(strand, oligo_c, salt_c)` group |
| `temperature` | temperature of the point [°C] |
| `signal` | measured fluorescence / absorbance |

`fluo_raw.csv.gz` is analysis-ready: the per-curve quality-control filtering and
the per-experiment cut/baseline fitting windows used in the paper are already
applied (windows are stored in `melting/variables.py`). The main result uses
only this file.

`uv_raw.csv.gz` has the same columns plus a `name` column (wavelength + scan
range, e.g. `250nm_5-90C`) that keeps each wavelength and melting/annealing
direction a distinct measurement;

