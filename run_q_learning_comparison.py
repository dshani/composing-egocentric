"""Entry point for comparing full SR training with ego-only Q-learning."""

from copy import deepcopy
from pathlib import Path

from parameters import get_parameter_parser, parameters
from helper_functions_ import make_directories
import run_model


def main():
    parser = get_parameter_parser()
    parser.add_argument("--include_lesion", action="store_true")
    args, _ = parser.parse_known_args()
    include_lesion = getattr(args, "include_lesion", False)
    for arg in vars(args):
        if arg == "include_lesion":
            continue
        value = getattr(args, arg)
        if value is not None:
            parameters[arg] = value

    seed_path = make_directories(
        seed=parameters.seed,
        pars=parameters,
        comparison="q_learning",
        job_id=parameters.job_id,
    )

    variants = ["full", "q_learning"]
    if include_lesion:
        variants.append("lesionLEC")

    for variant in variants:
        for subdir in ("save_dict", "worlds", "model"):
            (seed_path / variant / subdir).mkdir(parents=True, exist_ok=True)

    if not parameters.ego_q_learning:
        full_params = deepcopy(parameters)
        full_params.ego_q_learning = False
        run_model.train_agent(full_params, save=True, seed_path=seed_path / "full")

    q_learning_params = deepcopy(parameters)
    q_learning_params.ego_q_learning = True
    q_learning_params.egamma = 0
    q_learning_params.SR_lr_e = 0
    run_model.train_agent(q_learning_params, save=True, seed_path=seed_path / "q_learning")

    if include_lesion:
        lesion_params = deepcopy(parameters)
        lesion_params.lesionLEC = True
        lesion_params.ego_q_learning = False
        run_model.train_agent(lesion_params, save=True, seed_path=seed_path / "lesionLEC")

    print(f"Run path: {seed_path.parent}")
    print(f"Seed path: {seed_path}")
    print("Arguments:")
    print(parameters)


if __name__ == "__main__":
    main()
