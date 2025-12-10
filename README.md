## Requirements

* [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

All other dependencies are specified in `environment.yml` and `requirements.pip.txt`.

---

## Environment setup

Create and activate the environment:

```bash
micromamba env create -f environment.yml
micromamba activate new_ego
pip install -r requirements.pip.txt
```

---

## Quick Start

Train a single agent with paper-identical parameters:

```python run_experiments.py --seed=1 --worlds_type="'random_rooms7'"
```


This runs the full model (ego+allo) and lesioned model (allo-only) comparison for 5000 episodes with environment switches every 1000 episodes. Results are saved to `./Results/<date>/<run>/lesion/`.

---

## Reproducing paper figures

To generate all figures used in the paper, run:

```bash
bash bash_scripts/paper_figs.sh
```
