"""Run one of the three submitted MA-ICP case studies."""

import argparse
import os
from pathlib import Path

# Mission workers already parallelize the simulations.
for name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("payload", "formation", "tracking"))
    parser.add_argument("--output", type=Path, help="results directory for this case")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plots-only", action="store_true", help="draw figures from saved results")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="continue formation or tracking checkpoints",
    )
    args = parser.parse_args()
    output = args.output or Path("results") / args.case
    if args.plots_only:
        from maicp.visualization import paper_figures

        paper_figures(args.case, output)
        print(f"Figures saved under {(output / 'figures').resolve()}")
        return
    if output.exists() and not args.resume:
        parser.error(f"{output} exists. Choose a fresh --output directory.")
    if args.case == "payload":
        if args.resume:
            parser.error("Payload requires a fresh output directory.")
        from experiments.payload.experiment import run
        run(output)
    else:
        if args.case == "formation":
            from experiments.formation.experiment import run
        else:
            from experiments.tracking.experiment import run
        run(output, resume=args.resume)
    print(f"Results saved under {output.resolve()}")


if __name__ == "__main__":
    main()
