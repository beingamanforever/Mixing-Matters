import argparse
import json
from pathlib import Path

from .analysis import summarize, validate_negative, validate_order, validate_phase1
from .data import read_rows
from .download import NAME, download
from .io import read_jsonl
from .positive_control import run_control
from .run import (
    MODEL,
    SEED,
    generation_count,
    plan,
    plan_negative,
    plan_order,
    run_certify_negative,
    run_certify_order,
    run_tracer,
)


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

    check = commands.add_parser("certify-check")
    check.add_argument("results", type=Path)
    check.add_argument("--kind", choices=["negative", "order"], required=True)

    cert_neg = commands.add_parser("certify-negative")
    cert_neg.add_argument("--data", type=Path, default=Path("data") / NAME)
    cert_neg.add_argument("--output", type=Path, required=True)
    cert_neg.add_argument("--revision", required=True)
    cert_neg.add_argument("--positive-control", type=Path)
    cert_neg.add_argument("--n", type=int, default=200)
    cert_neg.add_argument("--dry-run", action="store_true")

    cert_ord = commands.add_parser("certify-order")
    cert_ord.add_argument("--data", type=Path, default=Path("data") / NAME)
    cert_ord.add_argument("--output", type=Path, required=True)
    cert_ord.add_argument("--revision", required=True)
    cert_ord.add_argument("--positive-control", type=Path)
    cert_ord.add_argument("--n", type=int, default=200)
    cert_ord.add_argument("--positions", type=int, nargs="+", default=(0, 4, 9))
    cert_ord.add_argument("--perms", type=int, default=3)
    cert_ord.add_argument("--dry-run", action="store_true")

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
    elif args.command == "certify-negative" and args.dry_run:
        rows = read_rows(args.data)
        work = plan_negative(rows, n=args.n)
        print(json.dumps({"model": MODEL, "seed": SEED, "generations": generation_count(work)}))
    elif args.command == "certify-negative":
        run_certify_negative(args.data, args.output, args.revision, args.positive_control, n=args.n)
    elif args.command == "certify-order" and args.dry_run:
        rows = read_rows(args.data)
        work = plan_order(rows, n=args.n, positions=tuple(args.positions), perms=args.perms)
        print(json.dumps({"model": MODEL, "seed": SEED, "generations": generation_count(work)}))
    elif args.command == "certify-order":
        run_certify_order(
            args.data,
            args.output,
            args.revision,
            args.positive_control,
            n=args.n,
            positions=tuple(args.positions),
            perms=args.perms,
        )
    elif args.command == "certify-check":
        records = read_jsonl(args.results)
        if args.kind == "negative":
            validate_negative(records)
            print("Negative control certification passed.")
        elif args.kind == "order":
            validate_order(records)
            print("Distractor order certification passed.")
    else:
        records = read_jsonl(args.results)
        validate_phase1(records)
        print(json.dumps(summarize(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
