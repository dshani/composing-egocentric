import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from figure_functions_ import generate_lesion_plot_


def load_accuracies(file_path):
    data = np.load(file_path, allow_pickle=True)["accuracies"]
    episodes = np.array([d[0] for d in data], dtype=float)
    steps = np.array([d[1] for d in data], dtype=float)
    return episodes, steps


def main():
    parser = argparse.ArgumentParser(description="Analyse nonadaptive comparison")
    parser.add_argument("--run_path", type=str, required=True, help="Path to comparison run")
    parser.add_argument("--sigma", type=float, default=5, help="Smoothing factor")
    parser.add_argument("--include_lesion", action="store_true", help="Include LEC lesion results")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    seed_dirs = sorted([d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("seed_")])

    full_curves = []
    nonadaptive_curves = []
    lesion_curves = []
    x_vals = None

    for seed in seed_dirs:
        episodes, steps_full = load_accuracies(seed / "full" / "save_dict" / "save_dict_accuracies.npz")
        _, steps_nonadaptive = load_accuracies(seed / "nonadaptive" / "save_dict" / "save_dict_accuracies.npz")
        if args.include_lesion:
            _, steps_lesion = load_accuracies(seed / "lesionLEC" / "save_dict" / "save_dict_accuracies.npz")
        if x_vals is None:
            x_vals = episodes
        full_curves.append(gaussian_filter1d(steps_full, args.sigma))
        nonadaptive_curves.append(gaussian_filter1d(steps_nonadaptive, args.sigma))
        if args.include_lesion:
            lesion_curves.append(gaussian_filter1d(steps_lesion, args.sigma))

    full_arr = np.vstack(full_curves)
    nonadaptive_arr = np.vstack(nonadaptive_curves)

    full_mean = np.mean(full_arr, axis=0)
    nonadaptive_mean = np.mean(nonadaptive_arr, axis=0)
    full_sem = np.std(full_arr, axis=0, ddof=1) / np.sqrt(full_arr.shape[0])
    nonadaptive_sem = np.std(nonadaptive_arr, axis=0, ddof=1) / np.sqrt(nonadaptive_arr.shape[0])

    inputs = [(full_mean, full_sem, x_vals), (nonadaptive_mean, nonadaptive_sem, x_vals)]
    labels = ["full agent", "nonadaptive agent"]

    if args.include_lesion and lesion_curves:
        lesion_arr = np.vstack(lesion_curves)
        lesion_mean = np.mean(lesion_arr, axis=0)
        lesion_sem = np.std(lesion_arr, axis=0, ddof=1) / np.sqrt(lesion_arr.shape[0])
        inputs.append((lesion_mean, lesion_sem, x_vals))
        labels.append("LEC lesion")

    fig, ax = plt.subplots()
    generate_lesion_plot_(
        ax,
        inputs=inputs,
        labels=labels,
    )

    results_dir = Path(__file__).resolve().parent / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "nonadaptive_lr_metrics.png"
    output_path_svg = results_dir / "nonadaptive_lr_metrics.svg"
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path_svg)
    plt.close(fig)
    print(f"Saved figure to {output_path}")
    print(f"Saved figure to {output_path_svg}")
    print(f"Full agent mean accuracy: {full_mean[-1]:.3f}")
    print(f"Nonadaptive agent mean accuracy: {nonadaptive_mean[-1]:.3f}")
    if args.include_lesion and lesion_curves:
        print(f"LEC lesion mean accuracy: {lesion_mean[-1]:.3f}")


if __name__ == "__main__":
    main()
