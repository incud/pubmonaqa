from pathlib import Path
import argparse

from monaqa2.data.single_step_quantum import launch_experiment_on_cineca


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["qemc", "coin"], required=True)
    parser.add_argument("--beta", type=float, required=True)
    args = parser.parse_args()

    launch_experiment_on_cineca(
        experiment=args.experiment,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
