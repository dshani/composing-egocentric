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

## Reproducing paper figures

To generate all figures used in the paper, run:

```bash
bash bash_scripts/paper_figs.sh
```

This will write the figures to the output directories expected by the manuscript.
