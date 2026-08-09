import argparse
import json
from pathlib import Path

from .analysis import summarize, validate_negative, validate_order, validate_phase1
from .audit import write_audit_sample
from .data import read_rows
from .download import NAME, download
from .figures import (
    write_figures,
    write_phase2_figures,
    write_phase4_figures,
    write_phase5_figures,
    write_phase6_figures,
    write_phase7_figures,
    write_phase8_figures,
)
from .io import read_jsonl
from .models import MODELS
from .positive_control import validate_control
from .run import (
    MODEL,
    NIAH_LENGTHS,
    SEED,
    generation_count,
    plan,
    plan_negative,
    plan_order,
    plan_sweep,
    run_certify_negative,
    run_certify_order,
    run_kv_control,
    run_ruler_sweep,
    run_sweep,
    run_tracer,
)


def _normalize_sweep_record(record: dict) -> dict:
    """Lift model_key out of software_versions, where run_sweep nests it.

    phase2.py expects model_key as a top-level field; run_sweep only writes
    it inside software_versions.
    """
    if "model_key" in record:
        return record
    model_key = record.get("software_versions", {}).get("model_key")
    if model_key is None:
        raise ValueError("record is missing model_key in software_versions")
    return {**record, "model_key": model_key}


def _print_control_outcome(model_key: str, control_path: Path) -> None:
    """Validate a key-value positive control file and report pass/fail.

    Never raises: a model failing key-value retrieval is a finding about the
    model, not a reason to abort the position sweep.
    """
    records = read_jsonl(control_path)
    if not records:
        print(f"Positive control for {model_key}: no records found in {control_path}")
        return
    fields = (
        "model",
        "model_revision",
        "seed",
        "python",
        "torch",
        "transformers",
        "cuda",
        "gpu",
        "attention_implementation",
    )
    metadata = {field: records[0].get(field) for field in fields}
    try:
        validate_control(control_path, metadata)
        print(f"Positive control for {model_key}: PASSED")
    except ValueError as error:
        print(f"Positive control for {model_key}: FAILED ({error})")


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
    control.add_argument(
        "--revision",
        help="Model tag or commit to resolve; defaults to the pinned registry revision",
    )
    control.add_argument("--model", choices=sorted(MODELS), default="pythia-2.8b")

    sweep_cmd = commands.add_parser("sweep")
    sweep_cmd.add_argument("--model", choices=sorted(MODELS), required=True)
    sweep_cmd.add_argument("--data", type=Path, default=Path("data") / NAME)
    sweep_cmd.add_argument("--output", type=Path, required=True)
    sweep_cmd.add_argument(
        "--revision",
        help="Model tag or commit to resolve; defaults to the pinned registry revision",
    )
    sweep_cmd.add_argument("--questions", type=int, default=800)
    sweep_cmd.add_argument(
        "--max-prompt-token-span",
        type=int,
        default=None,
        help=(
            "Tolerated span of prompt token counts across the ten gold positions of a "
            "question. Defaults to the module-level MAX_PROMPT_TOKEN_SPAN when omitted. "
            "Phase 8 raises this because the Llama, Qwen, and Nemotron-H tokenizers each "
            "produce different byte-pair merges at document boundaries."
        ),
    )
    sweep_cmd.add_argument("--positive-control", type=Path)
    sweep_cmd.add_argument("--dry-run", action="store_true")

    ruler_cmd = commands.add_parser("ruler-sweep")
    ruler_cmd.add_argument("--model", choices=sorted(MODELS), required=True)
    ruler_cmd.add_argument("--output", type=Path, required=True)
    ruler_cmd.add_argument(
        "--revision",
        help="Model tag or commit to resolve; defaults to the pinned registry revision",
    )
    ruler_cmd.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(NIAH_LENGTHS),
        help="Context length targets in tokens, generation budget included",
    )
    ruler_cmd.add_argument("--num-samples", type=int, default=100)
    ruler_cmd.add_argument("--dry-run", action="store_true")

    analyze = commands.add_parser("analyze")
    analyze.add_argument("results", type=Path)

    check = commands.add_parser("certify-check")
    check.add_argument("results", type=Path)
    check.add_argument("--kind", choices=["negative", "order"], required=True)

    figures_cmd = commands.add_parser("figures")
    figures_cmd.add_argument("--kv", type=Path, required=True)
    figures_cmd.add_argument("--phase1", type=Path, required=True)
    figures_cmd.add_argument("--output", type=Path, required=True)

    phase2_report = commands.add_parser("phase2-report")
    phase2_report.add_argument("--results", type=Path, nargs="+", required=True)
    phase2_report.add_argument("--output", type=Path, required=True)

    phase4_report = commands.add_parser("phase4-report")
    phase4_report.add_argument("--results", type=Path, nargs="+", required=True)
    phase4_report.add_argument("--output", type=Path, required=True)

    phase5_report = commands.add_parser("phase5-report")
    phase5_report.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Sweep files for the Pile and SlimPajama 2.8B Mamba models",
    )
    phase5_report.add_argument(
        "--architecture-results",
        type=Path,
        nargs="+",
        help="Optional Phase 2 Pythia and Pile-Mamba sweeps for the cross-phase comparison",
    )
    phase5_report.add_argument("--output", type=Path, required=True)

    phase6_report = commands.add_parser("phase6-report")
    phase6_report.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="niah_single_1 sweep files for the Phase 2 models",
    )
    phase6_report.add_argument(
        "--qa-results",
        type=Path,
        nargs="+",
        help="Optional Phase 2 QA sweeps for the niah-versus-QA edge comparison",
    )
    phase6_report.add_argument("--output", type=Path, required=True)

    phase7_report = commands.add_parser("phase7-report")
    phase7_report.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Sweep files spanning the Pythia depths, Phase 2, and any other model whose gold-condition records should join the mechanism analyses",
    )
    phase7_report.add_argument("--output", type=Path, required=True)

    phase8_report = commands.add_parser("phase8-report")
    phase8_report.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Sweep files for the three Phase 8 systems",
    )
    phase8_report.add_argument("--output", type=Path, required=True)

    audit_cmd = commands.add_parser("audit-sample")
    audit_cmd.add_argument("--results", type=Path, required=True)
    audit_cmd.add_argument("--output", type=Path, required=True)

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
        revision = args.revision or MODELS[args.model].revision
        run_kv_control(MODELS[args.model], args.output, revision)
    elif args.command == "sweep" and args.dry_run:
        rows = read_rows(args.data)
        work = plan_sweep(rows, questions=args.questions)
        print(json.dumps({"model": args.model, "seed": SEED, "generations": len(work)}))
    elif args.command == "sweep":
        revision = args.revision or MODELS[args.model].revision
        if args.positive_control is not None:
            _print_control_outcome(args.model, args.positive_control)
        sweep_kwargs = {"questions": args.questions}
        if args.max_prompt_token_span is not None:
            sweep_kwargs["max_prompt_token_span"] = args.max_prompt_token_span
        run_sweep(args.data, args.output, args.model, revision, **sweep_kwargs)
    elif args.command == "ruler-sweep" and args.dry_run:
        lengths = tuple(args.lengths)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "seed": SEED,
                    "lengths": list(lengths),
                    "num_samples": args.num_samples,
                    "generations": len(lengths) * args.num_samples * 12,
                }
            )
        )
    elif args.command == "ruler-sweep":
        revision = args.revision or MODELS[args.model].revision
        run_ruler_sweep(
            args.output,
            args.model,
            revision,
            lengths=tuple(args.lengths),
            num_samples=args.num_samples,
        )
    elif args.command == "phase6-report":
        records = [
            _normalize_sweep_record(record) for path in args.results for record in read_jsonl(path)
        ]
        qa_records = None
        if args.qa_results:
            qa_records = [
                _normalize_sweep_record(record)
                for path in args.qa_results
                for record in read_jsonl(path)
            ]
        paths = write_phase6_figures(records, args.output, qa_records)
        print(json.dumps({"paths": [str(path) for path in paths]}, indent=2))
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
    elif args.command == "figures":
        kv_records = read_jsonl(args.kv)
        phase1_records = read_jsonl(args.phase1)
        paths = write_figures(kv_records, phase1_records, args.output)
        print(json.dumps({"paths": [str(path) for path in paths]}, indent=2))
    elif args.command == "phase2-report":
        records = [
            _normalize_sweep_record(record) for path in args.results for record in read_jsonl(path)
        ]
        paths = write_phase2_figures(records, args.output)
        print(json.dumps({"paths": [str(path) for path in paths]}, indent=2))
    elif args.command == "phase4-report":
        records = [
            _normalize_sweep_record(record) for path in args.results for record in read_jsonl(path)
        ]
        paths = write_phase4_figures(records, args.output)
        print(json.dumps({"paths": [str(path) for path in paths]}, indent=2))
    elif args.command == "phase5-report":
        records = [
            _normalize_sweep_record(record) for path in args.results for record in read_jsonl(path)
        ]
        architecture_records = None
        if args.architecture_results:
            architecture_records = [
                _normalize_sweep_record(record)
                for path in args.architecture_results
                for record in read_jsonl(path)
            ]
        paths = write_phase5_figures(records, args.output, architecture_records)
        print(json.dumps({"paths": [str(path) for path in paths]}, indent=2))
    elif args.command == "phase7-report":
        records = [
            _normalize_sweep_record(record) for path in args.results for record in read_jsonl(path)
        ]
        paths = write_phase7_figures(records, args.output)
        print(json.dumps({"paths": [str(path) for path in paths]}, indent=2))
    elif args.command == "phase8-report":
        records = [
            _normalize_sweep_record(record) for path in args.results for record in read_jsonl(path)
        ]
        paths = write_phase8_figures(records, args.output)
        print(json.dumps({"paths": [str(path) for path in paths]}, indent=2))
    elif args.command == "audit-sample":
        records = read_jsonl(args.results)
        paths = write_audit_sample(records, args.output)
        print(json.dumps({"paths": [str(path) for path in paths]}, indent=2))
    else:
        records = read_jsonl(args.results)
        validate_phase1(records)
        print(json.dumps(summarize(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
