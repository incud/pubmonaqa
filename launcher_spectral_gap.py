from pathlib import Path
import argparse

from monaqa2.data.filename import MONAQA2_PARENT
from monaqa2.data.spectral_gap import run_experiment_to_generate_spectral_gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--idx-min", type=int, required=True)
    parser.add_argument("--idx-max", type=int, required=True)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()

    out_file = args.out_file
    if out_file is None:
        out_file = MONAQA2_PARENT / f"data/spectral_gaps_n{args.n}_idx{args.idx_min}to{args.idx_max}.pkl"

    run_experiment_to_generate_spectral_gaps(
        n_list=[args.n],
        idx_min=args.idx_min,
        idx_max=args.idx_max,
        out_file=out_file,
    )


if __name__ == "__main__":
    main()
