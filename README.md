## Requirements

* [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

All other dependencies are specified in `environment.yml` and `requirements.pip.txt`.

---

## Environment Setup

```bash
micromamba env create -f environment.yml
micromamba activate new_ego
pip install -r requirements.pip.txt
```

---

## Quick Start

```bash
python run_experiments.py --seed=1 --worlds_type="'random_rooms7'"
```

This trains the full model and lesioned model for comparison. Results are saved to `./Results/<date>/<run>/lesion/`.

---

## Reproducing Paper Figures

```bash
bash bash_scripts/paper_figs.sh
```

Figures are saved to `./figures/`.

---

## Output Structure

```
Results/<date>/<run>/lesion/seed_<N>/
├── unlesioned/       # Full model (ego+allo)
└── lesionLEC/        # Allocentric-only model
    ├── save_dict/    # Weight snapshots
    ├── worlds/       # Environment configs
    └── model/        # Trained model
```

---

## Interpreting Results

- **Average Steps**: Steps per episode to reach goal (lower = better)
- **unlesioned/**: Full model with egocentric + allocentric SRs
- **lesionLEC/**: Egocentric representations disabled

The unlesioned model should adapt faster after environment switches.
