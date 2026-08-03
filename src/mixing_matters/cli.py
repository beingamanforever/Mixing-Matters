import argparse
import json
from pathlib import Path

from .analysis import summarize, validate_phase1
from .data import read_rows
from .download import NAME, download
from .io import read_jsonl
from .positive_control import run_control
from .run import MODEL, SEED, plan, run_tracer


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--output", type=Path, default=Path("data") / NAME)
    run = commands.add_parser("run")
    run.add_argument("--data", type=Path, default=Path("data") / NAME)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--revision", required=True, help="Model tag or commit to resolve and record")
    run.add_argument("--positive-control", type=Path)
    run.add_argument("--dry-run", action="store_true")
    control = commands.add_parser("positive-control")
    control.add_argument("--output", type=Path, required=True)
    control.add_argument("--revision", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("results", type=Path)
    args = parser.parse_args()

    if args.command == "download":
        download(args.output)
    elif args.command == "run" and args.dry_run:
        rows = read_rows(args.data)
        print(json.dumps({"model": MODEL, "seed": SEED, "generations": len(plan(rows))}))
    elif args.command == "run":
        if args.positive_control is None:
            parser.error("run requires --positive-control unless --dry-run is set")
        run_tracer(args.data, args.output, args.revision, args.positive_control)
    elif args.command == "positive-control":
        run_control(args.output, args.revision)
    else:
        records = read_jsonl(args.results)
        validate_phase1(records)
        print(json.dumps(summarize(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
