"""Build the public release dataset from every committed sweep in ``artifacts/``.

The raw sweeps are stored per phase in shapes that grew with the experiments:
QA sweeps, key-value control sweeps, prompt-variant sweeps, and attention-sink
scans all carry different field names. This module flattens them into two
tabular record streams with one shared schema each, plus small aggregate CSVs
that describe what was released.

Prompt text is deliberately excluded. A ten-document prompt averages 5.5 KB, so
carrying it would multiply the release by roughly twenty times for text that the
harness rebuilds deterministically from ``question_id`` and ``gold_position``.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .models import MODELS

# The mixer axis the paper compares. ``family`` in the model registry is finer
# grained than this because it also selects an execution path.
MIXERS: dict[str, str] = {
    "pythia": "attention",
    "llama": "attention",
    "qwen2": "attention",
    "mamba": "state-space",
    "mamba2": "state-space",
    "mamba2-hybrid": "hybrid",
    "nemotron-h": "hybrid",
}

QA = "multidoc_qa"
KV = "kv_retrieval"
BASELINE = "liu_baseline"


@dataclass(frozen=True)
class Run:
    """One executed sweep file and the experimental conditions it was run under."""

    key: str
    phase: str
    experiment: str
    path: str
    model_key: str
    task: str = QA
    prompt_variant: str = BASELINE
    prompt_template: str = BASELINE


def _qa_runs() -> tuple[Run, ...]:
    scale = [
        ("130m-160m", "pythia-160m", "mamba-130m"),
        ("370m-410m", "pythia-410m", "mamba-370m"),
        ("790m-1b", "pythia-1b", "mamba-790m"),
        ("1.4b-1.4b", "pythia-1.4b", "mamba-1.4b"),
        ("2.8b-2.8b", "pythia-2.8b", "mamba-2.8b"),
    ]
    runs = [
        Run(
            key="phase1/pythia-2.8b",
            phase="phase1",
            experiment="calibration",
            path="artifacts/phase1/tracer.jsonl.gz",
            model_key="pythia-2.8b",
        )
    ]
    runs += [
        Run(
            key=f"phase2/{model}",
            phase="phase2",
            experiment="matched-2.8b-architecture",
            path=f"artifacts/phase2/{model}/sweep.jsonl.gz",
            model_key=model,
        )
        for model in ("pythia-2.8b", "mamba-2.8b", "mamba2-2.7b")
    ]
    runs += [
        Run(
            key=f"phase3/{model}",
            phase="phase3",
            experiment="matched-8b-pure-vs-hybrid",
            path=f"artifacts/phase3/{model}/sweep.jsonl",
            model_key=model,
        )
        for model in ("mamba2-8b", "mamba2-hybrid-8b")
    ]
    runs += [
        Run(
            key=f"phase4/{model}",
            phase="phase4",
            experiment=f"scale-pair-{pair}",
            path=f"artifacts/phase4/{pair}/{model}-sweep.jsonl.gz",
            model_key=model,
        )
        for pair, pythia, mamba in scale
        for model in (pythia, mamba)
    ]
    runs += [
        Run(
            key=f"phase5/{model}",
            phase="phase5",
            experiment="pretraining-corpus",
            path=f"artifacts/phase5/{model}/sweep.jsonl.gz",
            model_key=model,
        )
        for model in ("mamba-2.8b", "mamba-2.8b-slimpj")
    ]
    runs += [
        Run(
            key=f"phase7/{model}-{variant}",
            phase="phase7",
            experiment="query-position-variant",
            path=f"artifacts/phase7-mechanisms/4a-query-position/{model}-{variant}/sweep.jsonl.gz",
            model_key=model,
            prompt_variant=variant,
        )
        for model in ("pythia-2.8b", "mamba-2.8b")
        for variant in ("bookend", "question_first", "gold_padded")
    ]
    runs += [
        Run(
            key=f"phase7/{model}-tmpl-{template}",
            phase="phase7",
            experiment="instruction-template",
            path=f"artifacts/phase7-mechanisms/4e-template/{model}-{template}.jsonl.gz",
            model_key=model,
            prompt_template=template,
        )
        for model in ("pythia-2.8b", "mamba-2.8b")
        for template in ("concise", "instructional")
    ]
    runs += [
        Run(
            key="phase7/nemotron-h-8b-sink-blocked",
            phase="phase7",
            experiment="sink-block-intervention",
            path="artifacts/phase7-mechanisms/4c-sink-block/nemotron-h-8b-sink-blocked.jsonl.gz",
            model_key="nemotron-h-8b",
        )
    ]
    runs += [
        Run(
            key=f"phase8/{model}",
            phase="phase8",
            experiment="production-systems",
            path=f"artifacts/phase8/{model}/sweep.jsonl",
            model_key=model,
        )
        for model in ("nemotron-h-8b", "llama-3.1-8b", "qwen2.5-7b")
    ]
    return tuple(runs)


def _control_runs() -> tuple[Run, ...]:
    paths = {
        "phase1/pythia-2.8b": ("phase1", "artifacts/phase1/positive-control.jsonl.gz"),
        **{
            f"phase2/{model}": ("phase2", f"artifacts/phase2/{model}/positive-control.jsonl.gz")
            for model in ("pythia-2.8b", "mamba-2.8b", "mamba2-2.7b")
        },
        **{
            f"phase3/{model}": ("phase3", f"artifacts/phase3/{model}/positive-control.jsonl")
            for model in ("mamba2-8b", "mamba2-hybrid-8b")
        },
        **{
            f"phase4/{model}": (
                "phase4",
                f"artifacts/phase4/{pair}/{model}-positive-control.jsonl.gz",
            )
            for pair, models in {
                "130m-160m": ("pythia-160m", "mamba-130m"),
                "370m-410m": ("pythia-410m", "mamba-370m"),
                "790m-1b": ("pythia-1b", "mamba-790m"),
                "1.4b-1.4b": ("pythia-1.4b", "mamba-1.4b"),
                "2.8b-2.8b": ("pythia-2.8b", "mamba-2.8b"),
            }.items()
            for model in models
        },
        **{
            f"phase5/{model}": ("phase5", f"artifacts/phase5/{model}/positive-control.jsonl.gz")
            for model in ("mamba-2.8b", "mamba-2.8b-slimpj")
        },
        **{
            f"phase8/{model}": ("phase8", f"artifacts/phase8/{model}/positive-control.jsonl")
            for model in ("nemotron-h-8b", "llama-3.1-8b", "qwen2.5-7b")
        },
    }
    return tuple(
        Run(
            key=f"{key}-kv",
            phase=phase,
            experiment="key-value-positive-control",
            path=path,
            model_key=key.split("/", 1)[1],
            task=KV,
        )
        for key, (phase, path) in paths.items()
    )


def _sink_runs() -> tuple[Run, ...]:
    scan = "artifacts/phase7-mechanisms/4c-sink-scan"
    models = ("pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b")
    runs = [
        Run(
            key=f"phase7/sink-{model}",
            phase="phase7",
            experiment="attention-sink-scan",
            path=f"{scan}/{model}.jsonl.gz",
            model_key=model,
        )
        for model in models
    ]
    runs.append(
        Run(
            key="phase7/sink-pythia-2.8b",
            phase="phase7",
            experiment="attention-sink-scan",
            path=f"{scan}/pythia-2.8b-baseline.jsonl.gz",
            model_key="pythia-2.8b",
        )
    )
    runs += [
        Run(
            key=f"phase7/sink-pythia-2.8b-{variant}",
            phase="phase7",
            experiment="attention-sink-scan",
            path=f"{scan}-variants/pythia-2.8b-{variant}.jsonl.gz",
            model_key="pythia-2.8b",
            prompt_variant=variant,
        )
        for variant in ("bookend", "question_first")
    ]
    return tuple(runs)


GENERATION_RUNS: tuple[Run, ...] = _qa_runs() + _control_runs()
SINK_RUNS: tuple[Run, ...] = _sink_runs()

GENERATION_FIELDS: tuple[str, ...] = (
    "run_key",
    "phase",
    "experiment",
    "task",
    "model_key",
    "mixer",
    "family",
    "model_repo",
    "model_revision",
    "params_millions",
    "training_corpus",
    "prompt_variant",
    "prompt_template",
    "question_id",
    "source_index",
    "condition",
    "gold_position",
    "answers",
    "model_response",
    "score",
    "score_normalized_em",
    "score_first_line",
    "prompt_token_count",
    "generated_token_count",
    "gpu",
    "execution_path",
    "dtype",
    "run_id",
)

SINK_FIELDS: tuple[str, ...] = (
    "run_key",
    "phase",
    "model_key",
    "mixer",
    "family",
    "question_id",
    "source_index",
    "prompt_variant",
    "condition",
    "gold_position",
    "layer",
    "sink_mass",
    "prompt_token_count",
    "run_id",
)


def _read_lines(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def _model_columns(run: Run) -> dict:
    spec = MODELS[run.model_key]
    return {
        "run_key": run.key,
        "phase": run.phase,
        "experiment": run.experiment,
        "model_key": spec.key,
        "mixer": MIXERS[spec.family],
        "family": spec.family,
        "model_repo": spec.repo,
        "model_revision": spec.revision,
        "params_millions": spec.params_millions,
        "training_corpus": spec.training_corpus,
    }


def _kv_position(condition: str) -> int:
    return int(condition.rsplit("_", 1)[1])


def _variant(value: str | None, fallback: str) -> str:
    """Resolve a variant name, folding the sweeps' two names for the Liu prompt into one."""
    if value is None:
        return fallback
    return BASELINE if value == "baseline" else value


def _generation_row(run: Run, record: dict) -> dict:
    software = record.get("software_versions", record)
    if run.task == KV:
        condition = record["condition"]
        body = {
            "question_id": f"kv-{record['control_id']}",
            "source_index": record["control_id"],
            "condition": KV,
            "gold_position": _kv_position(condition),
            "answers": [record["gold"]],
            "model_response": record["generation"],
            "score_normalized_em": None,
            "score_first_line": None,
        }
    else:
        body = {
            "question_id": record["question_id"],
            "source_index": record.get("source_index"),
            "condition": record["condition"],
            "gold_position": record.get("gold_position"),
            "answers": record.get("answers"),
            "model_response": record.get("model_response"),
            "score_normalized_em": record.get("score_normalized_em"),
            "score_first_line": record.get("score_first_line"),
        }
    return {
        **_model_columns(run),
        "task": run.task,
        # Phase 7 stores the executed variant on the record itself, which is the
        # authority when a sweep mixes variants inside one file.
        "prompt_variant": _variant(record.get("prompt_variant"), run.prompt_variant),
        "prompt_template": _variant(record.get("prompt_template"), run.prompt_template),
        **body,
        "score": record.get("score"),
        "prompt_token_count": record.get("prompt_token_count"),
        "generated_token_count": record.get("generated_token_count"),
        "gpu": software.get("gpu"),
        "execution_path": software.get("execution_path"),
        "dtype": software.get("dtype"),
        "run_id": record.get("run_id"),
    }


def _sink_row(run: Run, record: dict) -> dict:
    columns = _model_columns(run)
    return {
        "run_key": run.key,
        "phase": run.phase,
        "model_key": columns["model_key"],
        "mixer": columns["mixer"],
        "family": columns["family"],
        "question_id": record["question_id"],
        "source_index": record.get("source_index"),
        "prompt_variant": _variant(record.get("prompt_variant"), run.prompt_variant),
        "condition": record.get("condition"),
        "gold_position": record.get("gold_position"),
        "layer": record["layer"],
        "sink_mass": record["sink_mass"],
        "prompt_token_count": record.get("prompt_token_count"),
        "run_id": record.get("run_id"),
    }


def iter_rows(root: Path, runs: tuple[Run, ...], row_of) -> Iterator[tuple[Run, dict]]:
    """Yield ``(run, row)`` for every run whose sweep file is present.

    A run with no file is skipped rather than raising, because ``artifacts/``
    is the committed subset of everything that was executed.
    """
    for run in runs:
        path = root / run.path
        if not path.exists():
            continue
        for record in _read_lines(path):
            yield run, row_of(run, record)


def iter_generations(root: Path) -> Iterator[dict]:
    """Yield every released generation row."""
    for _, row in iter_rows(root, GENERATION_RUNS, _generation_row):
        yield row


def iter_attention_sink(root: Path) -> Iterator[dict]:
    """Yield every released attention-sink row."""
    for _, row in iter_rows(root, SINK_RUNS, _sink_row):
        yield row


def _write_jsonl_gz(path: Path, rows: Iterator[dict], fields: tuple[str, ...]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # mtime=0 keeps repeated builds byte-identical for the same inputs.
    with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as raw:
        for row in rows:
            ordered = {field: row[field] for field in fields}
            raw.write((json.dumps(ordered, ensure_ascii=False) + "\n").encode())
            count += 1
    return count


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


RUN_FIELDS: tuple[str, ...] = (
    "run_key",
    "phase",
    "experiment",
    "task",
    "model_key",
    "mixer",
    "model_repo",
    "model_revision",
    "params_millions",
    "training_corpus",
    "prompt_variant",
    "prompt_template",
    "record_count",
    "question_count",
    "mean_score",
)

POSITION_FIELDS: tuple[str, ...] = ("run_key", "gold_position", "question_count", "accuracy")


class _Tally:
    """Per-run counters accumulated in the same pass that writes the records.

    The generation stream is large enough that a second decompression pass to
    build the aggregate CSVs would dominate the build, so counting happens while
    the rows stream past.
    """

    def __init__(self) -> None:
        self.order: list[tuple[Run, str]] = []
        self.records: dict[str, int] = defaultdict(int)
        self.questions: dict[str, set] = defaultdict(set)
        self.scores: dict[str, list] = defaultdict(lambda: [0, 0.0])
        self.positions: dict[tuple[str, int], list] = defaultdict(lambda: [0, 0.0])

    def start(self, run: Run, task: str) -> None:
        self.order.append((run, task))

    def add(self, run: Run, row: dict) -> None:
        self.records[run.key] += 1
        self.questions[run.key].add(row["question_id"])
        score = row.get("score")
        if score is None:
            return
        self.scores[run.key][0] += 1
        self.scores[run.key][1] += score
        position = row.get("gold_position")
        if position is not None:
            slot = self.positions[(run.key, position)]
            slot[0] += 1
            slot[1] += score

    def run_rows(self) -> list[dict]:
        rows = []
        for run, task in self.order:
            spec = MODELS[run.model_key]
            count, total = self.scores[run.key]
            rows.append(
                {
                    "run_key": run.key,
                    "phase": run.phase,
                    "experiment": run.experiment,
                    "task": task,
                    "model_key": spec.key,
                    "mixer": MIXERS[spec.family],
                    "model_repo": spec.repo,
                    "model_revision": spec.revision,
                    "params_millions": spec.params_millions,
                    "training_corpus": spec.training_corpus,
                    "prompt_variant": run.prompt_variant,
                    "prompt_template": run.prompt_template,
                    "record_count": self.records[run.key],
                    "question_count": len(self.questions[run.key]),
                    "mean_score": round(total / count, 6) if count else None,
                }
            )
        return rows

    def position_rows(self) -> list[dict]:
        rows = []
        for key in sorted(self.positions):
            count, total = self.positions[key]
            rows.append(
                {
                    "run_key": key[0],
                    "gold_position": key[1],
                    "question_count": count,
                    "accuracy": round(total / count, 6),
                }
            )
        return rows


def _tallied(root: Path, runs: tuple[Run, ...], row_of, task: str | None, tally: _Tally):
    seen = set()
    for run, row in iter_rows(root, runs, row_of):
        if run.key not in seen:
            seen.add(run.key)
            tally.start(run, task or run.task)
        tally.add(run, row)
        yield row


def build_dataset(root: Path, output: Path) -> dict[str, int]:
    """Write the release dataset and return the record count of each output."""
    tally = _Tally()
    counts = {
        "generations": _write_jsonl_gz(
            output / "generations.jsonl.gz",
            _tallied(root, GENERATION_RUNS, _generation_row, None, tally),
            GENERATION_FIELDS,
        ),
        "attention_sink": _write_jsonl_gz(
            output / "attention_sink.jsonl.gz",
            _tallied(root, SINK_RUNS, _sink_row, "attention_sink", tally),
            SINK_FIELDS,
        ),
    }
    run_rows = tally.run_rows()
    position_rows = tally.position_rows()
    _write_csv(output / "runs.csv", RUN_FIELDS, run_rows)
    _write_csv(output / "position_accuracy.csv", POSITION_FIELDS, position_rows)
    counts["runs"] = len(run_rows)
    counts["positions"] = len(position_rows)
    return counts
