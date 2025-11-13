"""
@author: Daniel Shani
"""

import datetime
import time
import functools
import logging
import operator
import os
from copy import deepcopy
from functools import reduce  # forward compatibility for Python 3
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
import tqdm
from natsort import natsorted
from scipy import linalg
from scipy.sparse.csgraph import laplacian
from tqdm import tqdm

import pickle


# def _merge_payload(existing, new):
#     """Merge ``new`` into ``existing`` while preserving nested structure."""

#     if isinstance(existing, dict) and isinstance(new, dict):
#         merged = dict(existing)
#         for sub_key, sub_value in new.items():
#             if sub_key in merged:
#                 merged[sub_key] = _merge_payload(merged[sub_key], sub_value)
#             else:
#                 merged[sub_key] = sub_value
#         return merged

#     if isinstance(existing, list) and isinstance(new, list):
#         return [*existing, *new]

#     return new


def _clean_save_dict_payload(payload):
    """Remove stray character keys introduced by malformed ``save_params``."""

    if isinstance(payload, dict):
        cleaned: dict = {}
        for raw_key, value in payload.items():
            key = raw_key
            if isinstance(raw_key, str):
                candidate = raw_key.strip()
                if not any(char.isalnum() for char in candidate):
                    # Skip keys that originate from iterating over raw strings
                    # such as "[" or "]".
                    continue
                key = candidate

            cleaned_value = _clean_save_dict_payload(value)
            if key in cleaned:
                cleaned[key] = _merge_payload(cleaned[key], cleaned_value)
            else:
                cleaned[key] = cleaned_value
        return cleaned

    if isinstance(payload, list):
        return [_clean_save_dict_payload(item) for item in payload]

    if isinstance(payload, tuple):
        return tuple(_clean_save_dict_payload(item) for item in payload)

    return payload


# def get_by_path(root, items):
#     """Access a nested object in root by item sequence."""
#     return reduce(operator.getitem, items, root)


# def set_by_path(root, items, value):
#     """Set a value in a nested object in root by item sequence."""
#     get_by_path(root, items[:-1])[items[-1]] = value


# def get_directory_structure_(p, must_contain=None, cant_contain=None):
#     # p = pathlib.Path(path)
#     return {x.name: get_directory_structure_(x, must_contain=must_contain, cant_contain=cant_contain) if x.is_dir() else x.name if x.name.endswith(".log") else np.load(x,
#                                                                                                                                                                         allow_pickle=True)
#             if (must_contain is None or any([y in x.name for y in must_contain])) and (
#                 cant_contain is None or not any([y in x.name for y in cant_contain])) else None
#             for x in p.iterdir()}
#
#
# def get_directory_structure(path, must_contain=None, cant_contain=None):
#     p = Path(path)
#     return get_directory_structure_(p, must_contain, cant_contain)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args) if hasattr(obj, attr) else None

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# def find_cone(x, y):
#     if y > 0:
#         G = [(y + 1 / 2) / (x + 1 / 2), (y - 1 / 2) / (x + 1 / 2),
#              (y + 1 / 2) / (x - 1 / 2), (y - 1 / 2) / (x - 1 / 2)]
#         full_cone = []
#         for g1 in G:
#             for g2 in G:
#                 cone_(g1, g2, full_cone)
#         return full_cone
#     else:
#         full_cone = []
#         if x > 0:
#             full_cone.append([0, (y + 1 / 2) / (x - 1 / 2)])
#         elif x < 0:
#             full_cone.append([(y + 1 / 2) / (x + 1 / 2), 0])
#         else:
#             return ValueError("x and y cannot be zero")
#         return full_cone


# def cone_(g1, g2, G):
#     if g1 == g2:
#         return [g1]
#     elif np.sign(g1) == np.sign(g2):
#         G.append([np.min([g1, g2]), np.max([g1, g2])])
#     else:
#         G.append([np.max([g1, g2]), np.inf])
#         G.append([-np.inf, np.min([g1, g2])])


# def check_inside(g_, G):
#     for g in G:
#         if g[0] <= g_ <= g[1]:
#             return True
#     return False


def check_inside_(p_, p):
    """
    Check if point p_ is blocked by the cone created by a barrier at p.
    """
    x_, y_ = p_
    x, y = p
    if x * x_ < 0:
        return False

    if x > 0:
        if (2 * y - 1) / (2 * x + 1) <= y_ / x_ <= (2 * y + 1) / (
                2 * x - 1):
            return True
        else:
            return False
    elif x < 0:
        if (2 * y + 1) / (2 * x + 1) <= y_ / x_ <= (2 * y - 1) / (
                2 * x - 1):
            return True
        else:
            return False
    else:
        if (y_ >= y) and ((y_ / x_ >= 2 * y - 1) or (y_ / x_ <= 1 - 2 * y)):
            return True
        else:
            return False


# def get_smoothest_eigenfunctions(flattened_transitions, k=None):
#     """
#     :param flattened_transitions: NxN transition matrix
#     :param k: if given gives the k smoothest eigenfunctions
#     :return eigenfunctions: returns an (kxN) array with each column
#     corresponding to an eigenvector
#             eigenvalues: corresponding eigenvalues
#     """
#     N = flattened_transitions.shape[0]
#     W = np.max(flattened_transitions, 1)
#     L = laplacian(W)
#     if k:
#         eigenvalues, eigenfunctions = linalg.eigh(
#             L, subset_by_index=[N - k,
#                                 N - 1])
#     else:
#         eigenvalues, eigenfunctions = linalg.eigh(L)
#     return eigenfunctions.T, eigenvalues


class DotDict(dict):
    """ DotDict is a dictionary that allows for dot notation access. """

    def __getattr__(*args):
        try:
            val = dict.__getitem__(*args)
            return DotDict(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError()

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def to_dict(data):
        """
        Recursively transforms a dict to a dotted dictionary
        """
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, DotDict):
                    data[k] = dict(v)
                    DotDict.to_dict(data[k])
                elif isinstance(v, list):
                    data[k] = [DotDict.to_dict(i) for i in v]
        elif isinstance(data, list):
            return [DotDict.to_dict(i) for i in data]
        else:
            return data
        return dict(data)


def make_directories(base_path='./Results/', seed=None, pars=None, comparison='lesion', job_id=None):
    """
    Creates directories for storing data during a model training run
    """
    
    base_path = Path(base_path)
        
    if not base_path.exists():
        base_path.mkdir()

    subdirs = ['save_dict', 'worlds', 'model']
    # subdirs = ['save_dict', 'worlds', 'model', 'models'] #trying to save models at certain intervals

    if comparison is None:
        comparison = 'run'

    if job_id is None:

        # Get current date for saving folder
        date = datetime.datetime.today().strftime('%Y-%m-%d')
        if not base_path.joinpath(date).exists():
            base_path.joinpath(date).mkdir()

        date_path = base_path.joinpath(date)

        run = find_most_recent(list(date_path.iterdir()), [comparison], recent=-1)[0]
        if run is None:
            run = 0
        else:
            run = int(run)

        run_path = date_path.joinpath(comparison + '_' + str(run))
        seed_path = run_path.joinpath('seed_' + str(seed))

        if not seed_path.exists():
            seed_path.mkdir(parents=True, exist_ok=True)
        else:
            run += 1
            run_path = date_path.joinpath(comparison + '_' + str(run))
            seed_path = run_path.joinpath('seed_' + str(seed))

    
    else:

        job_dir = Path(f'./Results/job_ids/{int(job_id) + 1}')
        job_dir.mkdir(parents=True, exist_ok=True)
        run_path_file = job_dir / 'run_path.txt'

        if not run_path_file.exists():
            date = datetime.datetime.today().strftime('%Y-%m-%d')
            if not base_path.joinpath(date).exists():
                base_path.joinpath(date).mkdir()

            date_path = base_path.joinpath(date)
            run = find_most_recent(list(date_path.iterdir()), [comparison], recent=-1)[0]
            if run is None:
                run = 0
            else:
                run = int(run)

            run_path = date_path.joinpath(comparison + '_' + str(run))
            seed_path = run_path.joinpath('seed_' + str(seed))

            if not seed_path.exists():
                seed_path.mkdir(parents=True, exist_ok=True)
            else:
                run += 1
                run_path = date_path.joinpath(comparison + '_' + str(run))
                seed_path = run_path.joinpath('seed_' + str(seed))

            with open(run_path_file, 'w') as f:
                f.write(str(run_path))
        else:
            with open(run_path_file, 'r') as f:
                run_path = Path(f.read())
            seed_path = run_path.joinpath('seed_' + str(seed))
    
    param_prefix = None
    if comparison == 'lesion':
        param_prefix = ['unlesioned', 'lesionLEC', 'lesionMEC']

    for prefix in (param_prefix if param_prefix is not None else ['']):
        for subdir in subdirs:
            seed_path.joinpath(prefix).joinpath(subdir).mkdir(parents=True, exist_ok=True)
            
            
    return seed_path
        


# def build_lesioned_unlesioned(structure, param='accuracies'):
#     # todo: check this works

#     unlesioned = []
#     lesioned = []

#     for seed in tqdm(structure.keys()):
#         unlesioned.append(np.copy(np.concatenate(structure[seed]['unlesioned'][param])))
#         lesioned.append(np.copy(np.concatenate(structure[seed]['lesionLEC'][param])))

#     return unlesioned, lesioned

import os
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from natsort import natsorted
import pickle


import os
import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_structure(
    run=None,
    date=None,
    seed=None,
    save_dirs=None,
    compare=None,
    dict_params=None,
    return_seed=False,
    seeds_path=None,
    lesionMEC=False,
    debug=False,
    load_worlds=True,
    max_workers=8,
    param_time_slices: Mapping[str, Sequence[float]] | None = None,
):
    """
    Load and aggregate simulation data from specified directories.

    Parameters:
    - run (str or int): Identifier for the run.
    - date (str): Date of the simulation in 'YYYY-MM-DD' format.
    - seed (int): Specific seed to load.
    - save_dirs (list of str or Path): Directories where data is saved.
    - compare (str): Comparison parameter ('lesion', 'gamma', etc.).
    - dict_params (list of str): Specific parameters to load from save_dict.
    - return_seed (bool): Whether to return the seed along with save_dict.
    - seeds_path (str or Path): Path to the seeds directory.
    - load_worlds (bool): Whether to load worlds data. Defaults to True.
    - max_workers (int): Maximum number of threads for parallel file loading. Defaults to 8.
    - param_time_slices (Mapping[str, Sequence[float]]): Optional mapping from
      parameter name to the timesteps of interest. When provided, only the
      snapshots closest to the specified timesteps are retained for those
      parameters, which can substantially reduce load times for large arrays.

    Returns:
    - save_dict (dict): Aggregated simulation data.
    - seed (optional): The seed used if return_seed is True.
    """
    
    
    if param_time_slices:
        param_time_slices = {
            str(key): [float(value) for value in times if value is not None]
            for key, times in param_time_slices.items()
            if times
        }
    else:
        param_time_slices = {}

    # Helper Functions
    def initialize_save_dict():
        if param_prefix is None:
            return {}
        return {prefix: {} for prefix in param_prefix}

    def get_sorted_dates(directory):
        dates = []
        for d in os.listdir(directory):
            try:
                datetime.datetime.strptime(d, '%Y-%m-%d')
                dates.append(d)
            except ValueError:
                continue
        return sorted(dates)

    def handle_unpickling_error(index, message):
        print(message)
        save_dict.pop(index, None)

    def load_file(file):
        try:
            save_data = np.load(file, allow_pickle=True)
            if isinstance(save_data, np.lib.npyio.NpzFile):
                payload = dict(save_data)
            else:
                payload = save_data.item()
        except (ValueError, IOError, EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading file {file}: {e}")
            return None

        return _clean_save_dict_payload(payload)

    def load_files_into_save_dict(file_list, index, prefix=""):
        # Remove unnecessary sorting if order doesn't matter
        # If order matters, replace 'sorted' with 'natsorted' or keep as is
        files = sorted(file_list)
        target = save_dict[index][prefix] if prefix else save_dict[index]
        keys_set = set(dict_params) if dict_params else None

        # Prepare lists for keys
        for file in files:
            save_data = load_file(file)
            if save_data is None:
                continue
            available_keys = save_data.keys()
            keys = available_keys if keys_set is None else [k for k in keys_set if k in available_keys]
            for key in keys:
                if key not in target:
                    target[key] = []
                target[key].append(save_data[key])

    def load_files_into_save_dict_parallel(file_list, index, prefix="", max_workers=8):
        files = sorted(file_list)
        target = save_dict[index][prefix] if prefix else save_dict[index]
        keys_set = set(dict_params) if dict_params else None
        results = [None] * len(files)

        class _SnapshotAccumulator:
            def __init__(self, target_times: Sequence[float]):
                self._targets = sorted(float(time) for time in target_times)
                self._best: dict[float, tuple[float, object] | None] = {
                    target_time: None for target_time in self._targets
                }

            def consider(self, x_val: float, y_val: object) -> None:
                for target_time in self._targets:
                    current = self._best[target_time]
                    if current is None or abs(x_val - target_time) < abs(current[0] - target_time):
                        self._best[target_time] = (x_val, y_val)

            def as_list(self) -> list[tuple[float, object]]:
                entries: list[tuple[float, object]] = []
                for target_time in self._targets:
                    candidate = self._best.get(target_time)
                    if candidate is not None:
                        entries.append(candidate)
                entries.sort(key=lambda pair: pair[0])
                return entries

        accumulators: dict[str, _SnapshotAccumulator] = {}

        def _iter_entries(sequence):
            if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes)):
                iterable = sequence
            else:
                iterable = [sequence]
            for position, value in enumerate(iterable):
                if value is None:
                    continue
                if (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes))
                    and len(value) >= 2
                ):
                    x_val, y_val = value[0], value[1]
                else:
                    x_val, y_val = position, value
                try:
                    scalar_x = float(np.asarray(x_val).item())
                except Exception:
                    try:
                        scalar_x = float(x_val)
                    except Exception:
                        continue
                try:
                    y_array = np.array(y_val, copy=True)
                except Exception:
                    y_array = deepcopy(y_val)
                yield scalar_x, y_array

        def _get_accumulator(param_name: str) -> _SnapshotAccumulator | None:
            if not param_time_slices:
                return None
            times = param_time_slices.get(param_name)
            if not times:
                return None
            accumulator = accumulators.get(param_name)
            if accumulator is None:
                accumulator = _SnapshotAccumulator(times)
                accumulators[param_name] = accumulator
            return accumulator

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(load_file, file): file_index
                for file_index, file in enumerate(files)
            }
            for future in as_completed(future_to_index):
                file_index = future_to_index[future]
                save_data = future.result()
                results[file_index] = save_data

        for save_data in results:
            if save_data is None:
                continue
            available_keys = save_data.keys()
            keys = available_keys if keys_set is None else [k for k in keys_set if k in available_keys]
            for key in keys:
                if key not in target:
                    target[key] = []
                entries = save_data[key]
                accumulator = _get_accumulator(key)
                if accumulator is None:
                    target[key].append(entries)
                else:
                    for x_val, y_val in _iter_entries(entries):
                        accumulator.consider(x_val, y_val)

        if accumulators:
            final_target = save_dict[index][prefix] if prefix else save_dict[index]
            for param_name, accumulator in accumulators.items():
                final_target[param_name] = [accumulator.as_list()]

    def load_worlds_into_save_dict(worlds_file, index, prefix=""):
        try:
            worlds = np.load(worlds_file, allow_pickle=True)
            target = save_dict[index][prefix] if prefix else save_dict[index]
            target['worlds'] = worlds
        except (ValueError, IOError, pickle.UnpicklingError) as e:
            handle_unpickling_error(index, f"Error loading worlds file: {e}")

    def find_most_recent(items, must_contain=None, recent=-1):
        filtered_items = [item for item in items if all(s in item for s in must_contain)]
        return sorted(filtered_items)[recent] if filtered_items else None

    # Initialization
    param_prefix = None
    if compare == 'lesion':
        if lesionMEC:
            param_prefix = ['unlesioned', 'lesionLEC', 'lesionMEC']
        else:
            param_prefix = ['unlesioned', 'lesionLEC']
    elif compare == 'gamma':
        raise NotImplementedError("Gamma comparison is not implemented.")

    dict_name = 'full_save_dict.npz' if dict_params is None else f"save_dict_{'_'.join(sorted(dict_params))}.npz"
    save_dict = {}

    if save_dirs is None:
        raise ValueError("The 'save_dirs' parameter must be provided.")

    for save_dir in save_dirs:
        save_dir = Path(save_dir)
        dates = get_sorted_dates(save_dir)

        if seed is None:
            if seeds_path is None:
                date = date or dates[-1]
                compare = compare or 'run'
                date_path = save_dir / date
                run_dirs = [d for d in os.listdir(date_path) if compare in d]
                run = run or find_most_recent(run_dirs, must_contain=[compare])
                seeds_path = date_path / f"{compare}_{run}"
            else:
                seeds_path = Path(seeds_path)

            if (seeds_path / dict_name).exists():
                save_data = np.load(seeds_path / dict_name, allow_pickle=True)
                if isinstance(save_data, np.lib.npyio.NpzFile):
                    save_dict = dict(save_data)
                else:
                    save_dict = save_data.item()

                if return_seed:
                    seed_list = sorted(seeds_path.iterdir())
                    return save_dict, seed_list[-1]
                return save_dict
            else:
                seed_list = sorted(seeds_path.iterdir())
        else:
            if seeds_path is None:
                date = date or dates[-1]
                compare = compare or 'run'
                date_path = save_dir / date
                run_dirs = [d for d in os.listdir(date_path) if compare in d]
                run = run or find_most_recent(run_dirs, must_contain=[compare])
                seeds_path = date_path / f"{compare}_{run}"
            else:
                seeds_path = Path(seeds_path)

            seed_list = [seeds_path / f"seed_{seed}"]
            
        if debug:
            seed_list = seed_list[:3] # Limit to 3 seeds for debugging

        # Processing Seeds
        for index, seed_path in enumerate(tqdm(seed_list, desc="Processing seeds", total=len(seed_list))):
            if not seed_path.is_dir():
                continue

            save_dict[index] = initialize_save_dict()

            for prefix in (param_prefix or ['']):
                current_prefix = prefix if prefix else ''
                dict_path = seed_path / current_prefix / 'save_dict'

                if dict_path.exists():
                    compact_loaded = False
                    if dict_params == ['accuracies']:
                        compact_file = dict_path / 'save_dict_accuracies.npz'
                        if compact_file.exists():
                            save_data = load_file(compact_file)
                            if save_data and 'accuracies' in save_data:
                                target = save_dict[index][current_prefix] if current_prefix else save_dict[index]
                                # Extra list layer maintains compatibility with get_parameter_values_
                                target['accuracies'] = [save_data['accuracies'].tolist()]
                                compact_loaded = True
                    if not compact_loaded:
                        file_list = [f for f in dict_path.iterdir() if f.name != 'save_dict_accuracies.npz']
                        if file_list:
                            # Use parallel loading
                            load_files_into_save_dict_parallel(file_list, index, prefix=current_prefix, max_workers=max_workers)

                            # Check if save_dict[index] still exists
                            if index not in save_dict:
                                print(f"Skipping further processing for index {index} due to previous error.")
                                break
                else:
                    print(f"save_dict directory does not exist at {dict_path}")

                # Optionally load worlds data if requested
                if load_worlds and index in save_dict:
                    worlds_file = seed_path / current_prefix / 'worlds' / 'worlds.sav'
                    if worlds_file.exists():
                        load_worlds_into_save_dict(worlds_file, index, prefix=current_prefix)
                    else:
                        print(f"worlds.sav file does not exist at {worlds_file}")
                elif index not in save_dict:
                    break  # Exit the prefix loop and proceed to the next seed

    if return_seed:
        return save_dict, seed_path
    return save_dict


# def load_structure(
#     run=None, date=None, seed=None, save_dirs=None, compare=None, dict_params=None, 
#     return_seed=False, seeds_path=None
# ):
#     """
#     Load and aggregate simulation data from specified directories.

#     Parameters:
#     - run (str or int): Identifier for the run.
#     - date (str): Date of the simulation in 'YYYY-MM-DD' format.
#     - seed (int): Specific seed to load.
#     - save_dirs (list of str or Path): Directories where data is saved.
#     - compare (str): Comparison parameter ('lesion', 'gamma', etc.).
#     - dict_params (list of str): Specific parameters to load from save_dict.
#     - return_seed (bool): Whether to return the seed along with save_dict.
#     - seeds_path (str or Path): Path to the seeds directory.

#     Returns:
#     - save_dict (dict): Aggregated simulation data.
#     - seed (optional): The seed used if return_seed is True.
#     """
#     # Helper Functions
#     def initialize_save_dict():
#         if param_prefix is None:
#             return {}
#         return {prefix: {} for prefix in param_prefix}

#     def get_sorted_dates(directory):
#         dates = []
#         for d in os.listdir(directory):
#             try:
#                 datetime.datetime.strptime(d, '%Y-%m-%d')
#                 dates.append(d)
#             except ValueError:
#                 continue
#         return sorted(dates)

#     def handle_unpickling_error(index, message):
#         print(message)
#         save_dict.pop(index, None)

#     def load_files_into_save_dict(file_list, index, prefix=""):
#         try:
#             for file in tqdm(natsorted(file_list), desc="Processing files", leave=False):
#                 save_data = np.load(file, allow_pickle=True)
#                 if isinstance(save_data, np.lib.npyio.NpzFile):
#                     save_data = dict(save_data)
#                 else:
#                     save_data = save_data.item()

#                 keys = save_data.keys() if dict_params is None else dict_params
#                 for key in keys:
#                     target = save_dict[index][prefix] if prefix else save_dict[index]
#                     if key not in target:
#                         target[key] = []
#                     target[key].append(deepcopy(save_data[key]))
#         except (ValueError, IOError, pickle.UnpicklingError) as e:
#             handle_unpickling_error(index, f"Error loading file: {e}")

#     def load_worlds_into_save_dict(worlds_file, index, prefix=""):
#         try:
#             worlds = np.load(worlds_file, allow_pickle=True)
#             target = save_dict[index][prefix] if prefix else save_dict[index]
#             target['worlds'] = worlds
#         except (ValueError, IOError, pickle.UnpicklingError) as e:
#             handle_unpickling_error(index, f"Error loading worlds file: {e}")

#     def find_most_recent(items, must_contain=None, recent=-1):
#         filtered_items = [item for item in items if all(s in item for s in must_contain)]
#         return sorted(filtered_items)[recent] if filtered_items else None

#     # Initialization
#     param_prefix = None
#     if compare == 'lesion':
#         param_prefix = ['unlesioned', 'lesionLEC', 'lesionMEC']
#     elif compare == 'gamma':
#         raise NotImplementedError("Gamma comparison is not implemented.")

#     dict_name = 'full_save_dict.npz' if dict_params is None else f"save_dict_{'_'.join(sorted(dict_params))}.npz"
#     save_dict = {}

#     if save_dirs is None:
#         raise ValueError("The 'save_dirs' parameter must be provided.")

#     for save_dir in save_dirs:
#         save_dir = Path(save_dir)
#         dates = get_sorted_dates(save_dir)

#         if seed is None:
#             if seeds_path is None:
#                 date = date or dates[-1]
#                 compare = compare or 'run'
#                 date_path = save_dir / date
#                 run_dirs = [d for d in os.listdir(date_path) if compare in d]
#                 run = run or find_most_recent(run_dirs, must_contain=[compare])
#                 seeds_path = date_path / f"{compare}_{run}"
#             else:
#                 seeds_path = Path(seeds_path)

#             if (seeds_path / dict_name).exists():
#                 save_data = np.load(seeds_path / dict_name, allow_pickle=True)
#                 if isinstance(save_data, np.lib.npyio.NpzFile):
#                     save_dict = dict(save_data)
#                 else:
#                     save_dict = save_data.item()

#                 if return_seed:
#                     seed_list = sorted(seeds_path.iterdir())
#                     return save_dict, seed_list[-1]
#                 return save_dict
#             else:
#                 seed_list = sorted(seeds_path.iterdir())
#         else:
#             if seeds_path is None:
#                 date = date or dates[-1]
#                 compare = compare or 'run'
#                 date_path = save_dir / date
#                 run_dirs = [d for d in os.listdir(date_path) if compare in d]
#                 run = run or find_most_recent(run_dirs, must_contain=[compare])
#                 seeds_path = date_path / f"{compare}_{run}"
#             else:
#                 seeds_path = Path(seeds_path)

#             seed_list = [seeds_path / f"seed_{seed}"]
#         seed_list = seed_list[:23] # Limit to 6 seeds for debugging
#         # Processing Seeds
#         for index, seed_path in tqdm(enumerate(seed_list), desc="Processing seeds", total=len(seed_list)):
#             if not seed_path.is_dir():
#                 continue

#             save_dict[index] = initialize_save_dict()

#             for prefix in (param_prefix or ['']):
#                 current_prefix = prefix if prefix else ''
#                 dict_path = seed_path / current_prefix / 'save_dict'

#                 if dict_path.exists():
#                     file_list = list(dict_path.iterdir())
#                     if file_list:
#                         load_files_into_save_dict(file_list, index, prefix=current_prefix)

#                         # Check if save_dict[index] still exists
#                         if index not in save_dict:
#                             print(f"Skipping further processing for index {index} due to previous error.")
#                             break
#                 else:
#                     print(f"save_dict directory does not exist at {dict_path}")

#                 # Before processing worlds_file, check if save_dict[index] still exists
#                 if index in save_dict:
#                     worlds_file = seed_path / current_prefix / 'worlds' / 'worlds.sav'
#                     if worlds_file.exists():
#                         load_worlds_into_save_dict(worlds_file, index, prefix=current_prefix)
#                     else:
#                         print(f"worlds.sav file does not exist at {worlds_file}")
#                 else:
#                     break  # Exit the prefix loop and proceed to the next seed

#     if return_seed:
#         return save_dict, seed
#     return save_dict




# def load_structure(run, date, seed, save_dirs, compare=None, dict_params=None, return_seed=False, seeds_path=None):
#     save = False  # haven't got this to work yet
#     save_at_end = False
#     param_prefix = None
#     if compare == 'lesion':
#         param_prefix = ['unlesioned', 'lesionLEC', 'lesionMEC']
#     elif compare == 'gamma':
#         return NotImplementedError

#     dict_name = 'full_save_dict.npz' if dict_params is None else f"save_dict_{'_'.join(sorted(dict_params))}.npz"
#     for save_dir in save_dirs:

#         dates = []
#         for d in os.listdir(save_dir):
#             try:
#                 datetime.datetime.strptime(d, '%Y-%m-%d')
#                 dates.append(d)
#             except ValueError:
#                 pass



#         # print(f"save dir: {save_dir}")
        

#         if seed is None:
#             if seeds_path is not None:
#                 seeds_path = seeds_path
#             else:
#                 date = date if date is not None else max(dates)
#                 compare = compare if compare is not None else 'run'
#                 run = run if run is not None else \
#                     find_most_recent(os.listdir(save_dir.joinpath(date)), must_contain=[compare],
#                                      recent=-1)[0]
#                 date_path = save_dir.joinpath(date)
#                 seeds_path = date_path.joinpath(compare + '_' + str(run))
#             if seeds_path.joinpath(dict_name).exists():
#                 save_dict = np.load(seeds_path.joinpath(dict_name), allow_pickle=True).item()
#                 if return_seed:
#                     seed_list = list(seeds_path.iterdir())
#                     seed_list.sort()
#                     return save_dict, seed_list[-1]
#                 else:
#                     return save_dict
#             else:
#                 if save:
#                     save_at_end = True
#                     print("Save at End")

#                 seeds = seeds_path.iterdir()
#                 if param_prefix is None:
#                     save_dict = {0: {}}
#                 else:
#                     save_dict = {0: {prefix: {} for prefix in param_prefix}}

#                 seed_list = list(seeds)
#                 seed_list.sort()
#         else:
#             if seeds_path is not None:
#                 seeds_path = seeds_path
#                 seed_list = [seeds_path.joinpath('seed_' + str(seed))]
#             else:
#                 date = date if date is not None else max(dates)
#                 compare = compare if compare is not None else 'run'
#                 run = run if run is not None else \
#                     find_most_recent(os.listdir(save_dir.joinpath(date)), must_contain=[compare],
#                                      recent=-1)[0]
#                 date_path = save_dir.joinpath(date)
#                 seeds_path = date_path.joinpath(compare + '_' + str(run))
#                 seed_list = [Path(save_dir).joinpath(date, compare + '_' + str(run), 'seed_' + str(seed))]

#             if param_prefix is None:
#                 save_dict = {0: {}}
#             else:
#                 save_dict = {0: {prefix: {} for prefix in param_prefix}}

#         for index, seed in tqdm(enumerate(seed_list)):
#             if not seed.is_dir():
#                 continue
#             else:
#                 if save_dict.get(index) is None:
#                     if param_prefix is None:
#                         save_dict[index] = {}
#                     else:
#                         save_dict[index] = {prefix: {} for prefix in param_prefix}

#                 for prefix in (param_prefix if param_prefix is not None else ['']):
#                     dict_path = seed / prefix / 'save_dict'
#                     if dict_path.exists():
#                         list_of_files = dict_path.iterdir()
#                         for file in tqdm(natsorted(list_of_files)):
#                             save_dict_ = np.load(file, allow_pickle=True).item()     

#                             for key in (save_dict_.keys() if dict_params is None else dict_params):
#                                 if prefix:
#                                     if save_dict[index][prefix].get(key) is None:
#                                         save_dict[index][prefix][key] = []
#                                         save_dict[index][prefix][key].append(deepcopy(save_dict_[key]))
#                                     else:
#                                         save_dict[index][prefix][key].append(deepcopy(save_dict_[key]))
#                                 else:
#                                     if save_dict[index].get(key) is None:
#                                         save_dict[index][key] = []
#                                         save_dict[index][key].append(deepcopy(save_dict_[key]))
#                                     else:
#                                         save_dict[index][key].append(deepcopy(save_dict_[key]))

#                     worlds_file = seed / prefix / 'worlds' / 'worlds.sav'
#                     if worlds_file.exists():
#                         worlds = np.load(worlds_file, allow_pickle=True)
#                         if prefix:
#                             save_dict[index][prefix]['worlds'] = worlds
#                         else:
#                             save_dict[index]['worlds'] = worlds

#     if save_at_end and save:
#         print("Saving")
#         make_keys_strings(save_dict)
#         np.savez(str(seeds_path.joinpath(dict_name)), **save_dict)

#     if return_seed:
#         return save_dict, seed
#     else:
#         return save_dict


# def make_keys_strings(dictionary):
#     for key in dictionary.keys():
#         if isinstance(key, int):
#             dictionary[str(key)] = dictionary.pop(key)
#         if isinstance(dictionary[key], dict):
#             make_keys_strings(dictionary[key])
#     return dictionary


# def load_parameter(param, run, date, seed, save_dirs, compare=None, max_workers=8):
#     save_dict = load_structure(run, date, seed, save_dirs, compare,
#                                dict_params=[param], max_workers=max_workers)
#     return save_dict


def load_model(seed, recent=-1, compare=None, load_params=None, worlds_index=0):
    

    param_prefix = None
    if compare == 'lesion':
        param_prefix = ['unlesioned', 'lesionLEC', 'lesionMEC']
    elif compare == 'gamma':
        return NotImplementedError

    for prefix in (param_prefix if param_prefix is not None else ['']):
        while not (seed / prefix / 'model' / 'model.sav').exists():
            seed_num = int(seed.name.split('_')[-1])
            seed_num += 1
            seed = seed.parent / ('seed_' + str(seed_num))
            if seed_num > 100:
                print("No model found")
                model = None
                return model
        model = np.load(seed / prefix / 'model' / 'model.sav', allow_pickle=True)
        break

    dicts = seed / prefix / 'save_dict'

    if dicts.exists():
        list_of_dicts_ = list(dicts.iterdir())
        dict_names = [x.name for x in list_of_dicts_]

        list_of_dicts = [x for _, x in natsorted(zip(dict_names, list_of_dicts_))]
        recent_dict = np.load(list_of_dicts[recent], allow_pickle=True).item()
        recent_dict = _clean_save_dict_payload(recent_dict)
        print(recent_dict.keys())
        for param in load_params if load_params is not None else recent_dict.keys():
            print("dict param: ", param)
            if rgetattr(model, param) is not None:
                rsetattr(model, param, recent_dict[param][-1][1])

        worlds_file = seed / prefix / 'worlds' / 'worlds.sav'
        if worlds_file.exists():
            worlds = np.load(worlds_file, allow_pickle=True)
            model.switch_world(worlds[worlds_index])

    else:
        print("No save dict found")
        model = None

    return model

# def load_models(seed, recent=-1, compare=None, load_params=None, worlds_index=0):
    
#     # same as load_model but will return model for each param_prefix
    

#     param_prefix = None
#     if compare == 'lesion':
#         param_prefix = ['unlesioned', 'lesionLEC']
#     elif compare == 'gamma':
#         return NotImplementedError
    
#     models = []

#     for prefix in (param_prefix if param_prefix is not None else ['']):
#         while not (seed / prefix / 'model' / 'model.sav').exists():
#             seed_num = int(seed.name.split('_')[-1])
#             seed_num += 1
#             seed = seed.parent / ('seed_' + str(seed_num))
#             if seed_num > 100:
#                 print("No model found")
#                 model = None
#                 break
#         model = np.load(seed / prefix / 'model' / 'model.sav', allow_pickle=True)
        

#         dicts = seed / prefix / 'save_dict'

#         if dicts.exists():
#             list_of_dicts_ = list(dicts.iterdir())
#             dict_names = [x.name for x in list_of_dicts_]

#             list_of_dicts = [x for _, x in natsorted(zip(dict_names, list_of_dicts_))]
#             recent_dict = np.load(list_of_dicts[recent], allow_pickle=True).item()
#             recent_dict = _clean_save_dict_payload(recent_dict)
#             print(recent_dict.keys())
#             for param in load_params if load_params is not None else recent_dict.keys():
#                 print("dict param: ", param)
#                 if rgetattr(model, param) is not None:
#                     rsetattr(model, param, recent_dict[param][-1][1])

#             worlds_file = seed / prefix / 'worlds' / 'worlds.sav'
#             if worlds_file.exists():
#                 worlds = np.load(worlds_file, allow_pickle=True)
#                 model.switch_world(worlds[worlds_index])

#         else:
#             print("No save dict found")
#             model = None
        
#         models.append(model)

#     return models

def load_recent_model(run, date, seed, save_dirs, recent=-1, compare=None,
                      dict_params=None, load_params=None, seeds_path=None, max_workers=8):
    """
    Find the path where the trained model weights are stored, and return the
    latest training iteration.
    Returns a dictionary for each seed, with each containing a key for each of the subdirectories
    and inside each subdirectory is a list of the files in that directory ordered by iteration number.
    """

    if dict_params is not None:
        if load_params is None:
            load_params = dict_params

    print("Loading Structure")


    save_dict, seed = load_structure(run, date, seed, save_dirs, compare,
                                     dict_params, return_seed=True,
                                     seeds_path=seeds_path,
                                     max_workers=max_workers)
    

    print("Loading Model")

    model = load_model(seed, recent, compare, load_params)

    return save_dict, model


# def get_param_value(structure, param, seed, recent=-1):
#     if seed is not None:
#         recent_iter, _ = find_most_recent(list(structure["weight"][seed].keys()))
#         return get_by_path(structure, param.split('.'))[seed][param.split('.')[-1] + '_' + str(recent_iter) + '.npy']
#     else:
#         recent_iter, _ = find_most_recent(list(structure["weight"].keys()))
#         return get_by_path(structure, param.split('.'))[param.split('.')[-1] + '_' + str(recent_iter) + '.npy']


def find_most_recent(
        file_list, must_contain=None, cant_contain=None,
        recent=-1):
    """
    Accepts a list of strings/Path files of format X_n[.Y optional], returns highest
    number n
    Each of the strings needs to contain one of must_contain and can't
    contain any of cant_contain
    """
    # Find all iteration numbers from file list where files match and sort them

    # if is list of Path files, change into text
    if file_list and isinstance(file_list[0], Path):
        file_list = [str(x) for x in file_list]

    iter_numbers = [int(str(x.split('.')[0]).split('_')[-1])
                    for x in file_list
                    if (True if cant_contain is None else not any(
            [y in x for y in cant_contain]))
                    and (True if must_contain is None else any(
            [y in x for y in must_contain]))]
    iter_numbers.sort()

    # Index is the latest iteration, or None if no iterations were found at all
    index = None if len(iter_numbers) == 0 else str(
        np.unique(iter_numbers)[recent])
    return index, np.unique(iter_numbers)


def make_logger(gen_path, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # create a file handler
    if isinstance(gen_path, Path):
        handler = logging.FileHandler(gen_path.joinpath(name + '.log'))
    else:
        handler = logging.FileHandler(gen_path + name + '.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


def make_Barrier(world, x, y, d):
    world[x - 2:x + 3, y - 2:y + 3] = 1
    world[x - 1:x + 2, y - 1:y + 2] = 0
    if d == 0:
        world[x - 1:x + 2, y - 2] = 0
    elif d == 1:
        world[x - 2, y - 1:y + 2] = 0
    elif d == 2:
        world[x - 1:x + 2, y + 2] = 0
    elif d == 3:
        world[x + 2, y - 1:y + 2] = 0
    else:
        raise ValueError("must be 0,1,2,3")


def make_barrier(world, x, y, d):
    """
    Make a barrier in the world
    """
    world[x - 1:x + 2, y - 1:y + 2] = 1
    world[x, y] = 0
    if d == 0:
        world[x, y - 1] = 0
    elif d == 1:
        world[x - 1, y] = 0
    elif d == 2:
        world[x, y + 1] = 0
    elif d == 3:
        world[x + 2, y] = 0
    else:
        raise ValueError("must be 0,1,2,3")


def plain_world_like(gridworld):
    plain_world = np.zeros_like(gridworld)
    plain_world[:, 0] = 1
    plain_world[:, -1] = 1
    plain_world[0, :] = 1
    plain_world[-1, :] = 1
    reward_loc = np.where(gridworld == -1)
    plain_world[reward_loc[0], reward_loc[1]] = -1
    return plain_world


