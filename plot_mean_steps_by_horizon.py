import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os


from typing import Optional, Tuple


def _find_accuracy_file_for_seed(seed_dir: Path, preferred_condition: Optional[str]) -> Optional[Path]:
    """Return path to save_dict_accuracies.npz for a single seed.

    Supports either flat layout: seed_X/save_dict/...
    or nested by condition: seed_X/<condition>/save_dict/...
    """
    # Flat layout first
    flat_path = seed_dir / 'save_dict' / 'save_dict_accuracies.npz'
    if flat_path.exists():
        return flat_path

    # Preferred condition layout
    if preferred_condition:
        preferred_path = seed_dir / preferred_condition / 'save_dict' / 'save_dict_accuracies.npz'
        if preferred_path.exists():
            return preferred_path

    # Fallback: search any immediate subdirectory that contains the file
    for sub in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
        candidate = sub / 'save_dict' / 'save_dict_accuracies.npz'
        if candidate.exists():
            return candidate
    return None


def load_mean_steps(run_path: Path, preferred_condition: Optional[str]) -> Optional[Tuple[float, float]]:
    """Return the mean final step count and its standard error across seeds.

    If runs are organized by lesion condition, the preferred condition is used
    when available. Otherwise, the first available condition per seed is used.
    """
    seed_steps = []
    for seed_dir in sorted(run_path.glob('seed_*')):
        acc_file = _find_accuracy_file_for_seed(seed_dir, preferred_condition)
        if acc_file is None:
            continue
        data = np.load(acc_file, allow_pickle=True)
        accuracies = data.get('accuracies')
        if accuracies is None or len(accuracies) == 0:
            continue
        # accuracies is an array of (episode, step_count) tuples
        arr = np.array(accuracies, dtype=object)
        step_counts = [float(item[1]) for item in arr]
        seed_steps.append(step_counts[-1])
    if seed_steps:
        mean = float(np.mean(seed_steps))
        if len(seed_steps) > 1:
            err = float(np.std(seed_steps, ddof=1) / np.sqrt(len(seed_steps)))
        else:
            err = 0.0
        return mean, err
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Plot mean steps with error bars across horizons for a given worlds type')
    parser.add_argument('--worlds_type', required=True,
                        help='World type identifier, e.g., "random_rooms7"')
    parser.add_argument('--results_dir', default='Results/job_ids',
                        help='Directory containing job result subdirectories')
    parser.add_argument('--horizons', nargs='+', type=int,
                        help='Optional list of horizon values to include. If provided, the script will find only the most recent matching run for each horizon.')
    parser.add_argument('--condition', default='unlesioned',
                        help='Condition subfolder to use per seed (e.g., unlesioned, lesionLEC, lesionMEC). "unlesioned" by default; will fall back to any if missing.')
    parser.add_argument('--lesion_condition', default='lesionLEC',
                        help='Condition used for lesion baseline; plotted as a horizontal dotted line')
    parser.add_argument('--output',
                        help='Output path for the generated figure with error bars')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        # Fallback: if running from repo root, results may be under ego_release/bash_scripts/Results/job_ids
        script_dir = Path(__file__).resolve().parent
        candidate = script_dir / 'bash_scripts' / 'Results' / 'job_ids'
        if candidate.exists():
            results_dir = candidate
        else:
            raise FileNotFoundError(f'Results directory not found: {results_dir}')

    horizon_stats = {}
    lesion_means = []

    # Helper to iterate job dirs in reverse chronological order by numeric name
    def iter_job_dirs_newest_first(base: Path):
        numeric_dirs = []
        other_dirs = []
        for p in base.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if name.isdigit():
                numeric_dirs.append((int(name), p))
            else:
                other_dirs.append(p)
        # Newest first for numeric names; append non-numeric afterwards
        for _, p in sorted(numeric_dirs, key=lambda x: x[0], reverse=True):
            yield p
        for p in other_dirs:
            yield p

    if args.horizons:
        requested = sorted(set(args.horizons))
        latest_for_horizon = {}
        for subdir in iter_job_dirs_newest_first(results_dir):
            if len(latest_for_horizon) == len(requested):
                break
            wt_file = subdir / 'worlds_type.txt'
            horizon_file = subdir / 'horizon.txt'
            run_path_file = subdir / 'run_path.txt'
            if not (wt_file.exists() and horizon_file.exists() and run_path_file.exists()):
                continue
            wt = wt_file.read_text().strip()
            if wt != args.worlds_type:
                continue
            try:
                horizon = int(horizon_file.read_text().strip())
            except ValueError:
                continue
            if horizon not in requested or horizon in latest_for_horizon:
                continue
            latest_for_horizon[horizon] = subdir

        for horizon in requested:
            subdir = latest_for_horizon.get(horizon)
            if not subdir:
                continue
            run_path_file = subdir / 'run_path.txt'
            run_path = Path(run_path_file.read_text().strip())
            mean_and_err = load_mean_steps(run_path, args.condition)
            if mean_and_err is not None:
                horizon_stats[horizon] = mean_and_err
            lesion_mean_and_err = load_mean_steps(run_path, args.lesion_condition)
            if lesion_mean_and_err is not None:
                lesion_means.append(lesion_mean_and_err[0])
    else:
        # Original behavior: scan all matching entries
        for subdir in results_dir.iterdir():
            if not subdir.is_dir():
                continue
            wt_file = subdir / 'worlds_type.txt'
            horizon_file = subdir / 'horizon.txt'
            run_path_file = subdir / 'run_path.txt'
            if not (wt_file.exists() and horizon_file.exists() and run_path_file.exists()):
                continue
            wt = wt_file.read_text().strip()
            if wt != args.worlds_type:
                continue
            try:
                horizon = int(horizon_file.read_text().strip())
            except ValueError:
                continue
            run_path = Path(run_path_file.read_text().strip())
            mean_and_err = load_mean_steps(run_path, args.condition)
            if mean_and_err is not None:
                horizon_stats[horizon] = mean_and_err
            lesion_mean_and_err = load_mean_steps(run_path, args.lesion_condition)
            if lesion_mean_and_err is not None:
                lesion_means.append(lesion_mean_and_err[0])

    if not horizon_stats:
        print('No matching results found.')
        return

    lesion_mean = np.mean(lesion_means) if lesion_means else None
    horizons = sorted(horizon_stats.keys())
    means = [horizon_stats[h][0] for h in horizons]
    errors = [horizon_stats[h][1] for h in horizons]

    fig, ax = plt.subplots()
    ax.errorbar(horizons, means, yerr=errors, marker='o', label=args.condition)
    if lesion_mean is not None:
        ax.axhline(lesion_mean, linestyle=':', color='gray', label=args.lesion_condition)
    ax.legend()
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Mean Steps ± SEM')
    ax.set_title(args.worlds_type)
    output_path_svg = None

    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = output_path.resolve()
    else:
        # Save figures under bash_scripts/Results by default to match sbatch conventions
        script_dir = Path(__file__).resolve().parent
        bash_results_dir = script_dir / 'bash_scripts' / 'Results'
        output_path = bash_results_dir / f'mean_steps_by_horizon_{args.worlds_type}.png'
        output_path_svg = bash_results_dir / f'mean_steps_by_horizon_{args.worlds_type}.svg'

    os.makedirs(output_path.parent, exist_ok=True)
    fig.savefig(output_path)
    if output_path_svg:
        fig.savefig(output_path_svg)
    plt.close(fig)
    
    print(f'Saved figure to {output_path}')
    if output_path_svg:
        print(f'Saved figure to {output_path_svg}')


if __name__ == '__main__':
    main()
