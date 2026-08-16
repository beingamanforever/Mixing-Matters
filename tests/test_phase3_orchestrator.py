"""Tests for the immutable Phase 3 bare-metal orchestrator."""

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
PHASE3_SCRIPT = REPOSITORY_ROOT / "scripts" / "phase3.sh"
SETUP_SCRIPT = REPOSITORY_ROOT / "scripts" / "setup_baremetal_megatron.sh"
DATA_SHA256 = "192a05b27af2b09eec33ca0c94bb5cf82bcaf70d78b3bdff1258df34bf37aab9"
PIQA_SHA256 = "61533005e22f175534909b1ec8eacb6da03c233933558d1bafb15787453b1f55"
MEGATRON_SHA = "df61e60bf5670b1196fcae2264311401d3bb82db"
MODEL_REVISIONS = {
    "mamba2-8b": "b915550c63ba9359f88f44d1f6a600d85af27302",
    "mamba2-hybrid-8b": "35e8852e2240b350ac2fe2a3b8aa341b5930018e",
}
PURE_WEIGHT_SHA256 = "47c2766f6aad89d73beafbeaecb334aab902d7370906d081764a90bb7a8bbbcb"
HYBRID_WEIGHT_SHA256 = "480964a9dca70e3dcbf348bbf812ed2388476c2b2968cb903cca666d08547399"
TOKENIZER_SHA256 = "5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09"
LATEST_SHA256 = "3933e3274b63dc01fd286d6d939a626867c83a4603fc4347e0f8b8856f1b98fd"
LAUNCHER_SHA256 = "1" * 64


def test_setup_uses_venv_startup_shim_without_modifying_megatron() -> None:
    """The Python 3.12 workaround must live in the venv, not Megatron."""
    script = SETUP_SCRIPT.read_text()

    assert "mixing_torch_compat.pth" in script
    assert "import mixing_torch_compat" in script
    assert 'cat > "$SITE_PACKAGES/amp_C.py"' in script
    assert "from megatron.inference import text_generation" in script
    assert 'PATH="$VENV/bin:$PATH" make -C "$MEGATRON/megatron/core/datasets"' in script
    assert 'cat > "$MEGATRON/megatron/core/jit.py"' not in script


def test_launcher_uses_venv_build_tools() -> None:
    """Megatron's runtime dataset build must use the pinned venv dependencies."""
    launcher = (REPOSITORY_ROOT / "scripts" / "run_phase3_baremetal.sh").read_text()

    assert 'export PATH="$(dirname "${VENV}"):${PATH}"' in launcher


@pytest.mark.parametrize(
    ("environment_update", "message"),
    [
        ({"MIXING_GIT_STATUS": " M README.md"}, "Mixing-Matters checkout is dirty"),
        (
            {"MIXING_GIT_STATUS": " M scripts/run_phase3_baremetal.sh"},
            "Mixing-Matters checkout is dirty",
        ),
        ({"MEGATRON_GIT_STATUS": "?? scratch"}, "Megatron checkout is dirty"),
        ({"MEGATRON_GIT_SHA": "bad"}, "Megatron checkout must be"),
    ],
)
def test_phase3_refuses_unpinned_checkouts(
    tmp_path: Path,
    environment_update: dict[str, str],
    message: str,
) -> None:
    """Repository provenance guards must stop before creating artifacts."""
    environment, run_dir = _phase3_environment(tmp_path)
    environment.update(environment_update)

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert message in result.stderr
    assert not run_dir.exists()


def test_phase3_refuses_existing_run_directory(tmp_path: Path) -> None:
    """A run directory is an immutable experiment boundary."""
    environment, run_dir = _phase3_environment(tmp_path)
    run_dir.mkdir()

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert "run directory already exists" in result.stderr


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MM_PIQA_SAMPLES", "0", "MM_PIQA_SAMPLES must be 1838"),
        ("MM_PIQA_SAMPLES", "1", "MM_PIQA_SAMPLES must be 1838"),
        ("MM_PIQA_SAMPLES", "1837", "MM_PIQA_SAMPLES must be 1838"),
        ("MM_QUESTIONS", "0", "MM_QUESTIONS must be 800"),
        ("MM_QUESTIONS", "1", "MM_QUESTIONS must be 800"),
        ("MM_QUESTIONS", "799", "MM_QUESTIONS must be 800"),
    ],
)
def test_phase3_refuses_noncanonical_sample_counts(
    tmp_path: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    """Reduced or empty runs must not pass as the canonical experiment."""
    environment, run_dir = _phase3_environment(tmp_path)
    environment[name] = value

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert message in result.stderr
    assert not run_dir.exists()


def test_phase3_refuses_external_launcher(tmp_path: Path) -> None:
    """Only the launcher inside the clean Mixing-Matters checkout is valid."""
    environment, run_dir = _phase3_environment(tmp_path)
    external_launcher = tmp_path / "external-launcher.sh"
    _write_executable(external_launcher, "#!/usr/bin/env bash\nexit 0\n")
    environment["LAUNCHER"] = str(external_launcher)

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert "launcher must be the canonical repository launcher" in result.stderr
    assert not run_dir.exists()


def test_phase3_refuses_wrong_dataset_hash(tmp_path: Path) -> None:
    """The launcher must not run against an unpinned dataset."""
    environment, run_dir = _phase3_environment(tmp_path)
    environment["DATA_SHA256"] = "0" * 64

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert "dataset SHA-256 mismatch" in result.stderr
    assert not run_dir.exists()


def test_phase3_refuses_wrong_piqa_hash(tmp_path: Path) -> None:
    """The benchmark gate must use the pinned labeled PIQA split."""
    environment, run_dir = _phase3_environment(tmp_path)
    environment["PIQA_SHA256"] = "0" * 64

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert "PIQA SHA-256 mismatch" in result.stderr
    assert not run_dir.exists()


@pytest.mark.parametrize(
    "missing_input",
    [
        "piqa",
        "mamba2-8b/release/mp_rank_00/model_optim_rng.pt",
        "mamba2-hybrid-8b/latest_checkpointed_iteration.txt",
        "mamba2-8b/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model",
    ],
)
def test_phase3_refuses_missing_required_inputs(tmp_path: Path, missing_input: str) -> None:
    """PIQA and both checkpoint layouts must be complete before a run."""
    environment, run_dir = _phase3_environment(tmp_path)
    if missing_input == "piqa":
        missing_path = Path(environment["PIQA"])
    else:
        missing_path = Path(environment["CKPT_ROOT"], missing_input)
    missing_path.unlink()

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert f"required file not found: {missing_path}" in result.stderr
    assert not run_dir.exists()


def test_phase3_stops_when_piqa_validation_is_outside_gate(tmp_path: Path) -> None:
    """Both PIQA gates must pass before any control or sweep starts."""
    environment, run_dir = _phase3_environment(tmp_path)
    environment["FAIL_VALIDATION_MODEL"] = "mamba2-hybrid-8b"

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert "PIQA validation did not pass for mamba2-hybrid-8b" in result.stderr
    assert Path(environment["CALL_LOG"]).read_text().splitlines() == [
        "mamba2-8b validate",
        "mamba2-hybrid-8b validate",
    ]
    assert (run_dir / "mamba2-hybrid-8b-validate.log").is_file()


def test_phase3_refuses_wrong_checkpoint_hash(tmp_path: Path) -> None:
    """The checkpoint bytes must match the pinned upstream snapshot."""
    environment, run_dir = _phase3_environment(tmp_path)
    environment["FAKE_PURE_WEIGHT_SHA256"] = "0" * 64

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr
    assert "mamba2-8b/release/mp_rank_00/model_optim_rng.pt" in result.stderr
    assert not run_dir.exists()


@pytest.mark.parametrize(
    ("bad_output", "message"),
    [
        ("kv-malformed", "invalid JSONL"),
        ("kv-duplicate", "KV conditions are duplicated or incomplete"),
        ("sweep-incomplete", "expected 9600 sweep records, found 9599"),
        ("sweep-null", "has invalid score: None"),
        ("sweep-nan", "has invalid score: nan"),
        ("sweep-failures", "scoring-failure sidecar exists"),
    ],
)
def test_phase3_refuses_invalid_stage_artifacts(
    tmp_path: Path,
    bad_output: str,
    message: str,
) -> None:
    """A malformed, duplicate, incomplete, or failed stage must stop the run."""
    environment, run_dir = _phase3_environment(tmp_path)
    environment["BAD_OUTPUT"] = bad_output

    result = _run_phase3(run_dir, environment)

    assert result.returncode != 0
    assert message in result.stderr
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert "mamba2-hybrid-8b kv" not in calls


def test_phase3_writes_manifest_and_runs_stages_in_order(tmp_path: Path) -> None:
    """A clean run records provenance before executing all six stages."""
    environment, run_dir = _phase3_environment(tmp_path)

    result = _run_phase3(run_dir, environment)

    assert result.returncode == 0, result.stderr
    assert (run_dir / "mamba2-8b-validate.log").is_file()
    assert (run_dir / "mamba2-hybrid-8b-validate.log").is_file()
    calls = Path(environment["CALL_LOG"]).read_text().splitlines()
    assert calls == [
        "mamba2-8b validate",
        "mamba2-hybrid-8b validate",
        "mamba2-8b kv",
        "mamba2-8b sweep",
        "mamba2-hybrid-8b kv",
        "mamba2-hybrid-8b sweep",
    ]

    manifest = json.loads((run_dir / "environment.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["repositories"]["mixing_matters"]["status"] == []
    assert manifest["repositories"]["megatron"]["sha"] == MEGATRON_SHA
    assert manifest["repositories"]["megatron"]["status"] == []
    assert manifest["dataset"]["sha256"] == DATA_SHA256
    assert manifest["piqa"]["sha256"] == PIQA_SHA256
    assert manifest["packages"]
    assert manifest["nvidia_smi"] == "NVIDIA-SMI test fixture"
    assert manifest["checkpoint_revisions"] == MODEL_REVISIONS
    assert manifest["configuration"]["launcher_sha256"] == LAUNCHER_SHA256
    assert manifest["configuration"]["piqa_samples"] == 1838
    assert manifest["configuration"]["questions"] == 800
    assert manifest["checkpoint_files"]["mamba2-8b"]["model_optim_rng.pt"] == (PURE_WEIGHT_SHA256)
    assert manifest["checkpoint_files"]["mamba2-hybrid-8b"]["model_optim_rng.pt"] == (
        HYBRID_WEIGHT_SHA256
    )
    assert [stage["stage"] for stage in manifest["command_stages"]] == [
        "validate",
        "validate",
        "kv",
        "sweep",
        "kv",
        "sweep",
    ]


def _run_phase3(run_dir: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the orchestrator with a controlled host interface."""
    return subprocess.run(
        ["bash", str(PHASE3_SCRIPT), str(run_dir)],
        capture_output=True,
        env=environment,
        text=True,
        check=False,
    )


def _phase3_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Create the files and command shims needed for a CPU-only Phase 3 test."""
    mixing = tmp_path / "mixing"
    megatron = tmp_path / "Megatron-LM"
    checkpoint_root = tmp_path / "checkpoints"
    fake_bin = tmp_path / "bin"
    mixing.mkdir()
    megatron.mkdir()
    fake_bin.mkdir()

    data_directory = mixing / "data"
    data_directory.mkdir()
    data = data_directory / "nq-open-10_total_documents_gold_at_0.jsonl.gz"
    piqa = data_directory / "piqa_valid.jsonl"
    data.write_bytes(b"test dataset")
    piqa.write_text("{}\n")
    _create_checkpoints(checkpoint_root)

    call_log = tmp_path / "calls.log"
    scripts_directory = mixing / "scripts"
    scripts_directory.mkdir()
    launcher = scripts_directory / "run_phase3_baremetal.sh"
    _write_executable(
        launcher,
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s %s\\n' "$1" "$2" >> "$CALL_LOG"
if [ "$2" = validate ]; then
  if [ "$1" = mamba2-8b ]; then
    target=79.82
  else
    target=79.65
  fi
  verdict=PASS
  if [ "${FAIL_VALIDATION_MODEL:-}" = "$1" ]; then
    verdict='OUTSIDE +/-1.0'
  fi
  printf 'PIQA acc=%s target=%s delta=+0.00 [%s]\\n' \
    "$target" "$target" "$verdict"
else
  "$VENV" "$ARTIFACT_WRITER" "$1" "$2"
fi
""",
    )
    artifact_writer = tmp_path / "write_artifacts.py"
    _write_artifact_writer(artifact_writer)
    _write_fake_commands(fake_bin)

    environment = os.environ.copy()
    environment.update(
        {
            "CALL_LOG": str(call_log),
            "ARTIFACT_WRITER": str(artifact_writer),
            "CKPT_ROOT": str(checkpoint_root),
            "DATA": str(data),
            "DATA_SHA256": DATA_SHA256,
            "LAUNCHER": str(launcher),
            "MEGATRON": str(megatron),
            "MEGATRON_GIT_SHA": MEGATRON_SHA,
            "MEGATRON_GIT_STATUS": "",
            "MIXING": str(mixing),
            "MIXING_GIT_SHA": "a" * 40,
            "MIXING_GIT_STATUS": "",
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "PIQA": str(piqa),
            "PIQA_SHA256": PIQA_SHA256,
            "FAKE_PURE_WEIGHT_SHA256": PURE_WEIGHT_SHA256,
            "FAKE_HYBRID_WEIGHT_SHA256": HYBRID_WEIGHT_SHA256,
            "FAKE_TOKENIZER_SHA256": TOKENIZER_SHA256,
            "FAKE_LATEST_SHA256": LATEST_SHA256,
            "FAKE_LAUNCHER_SHA256": LAUNCHER_SHA256,
            "REAL_GIT": shutil.which("git", path=os.defpath) or "/usr/bin/git",
            "VENV": sys.executable,
        }
    )
    return environment, tmp_path / "run"


def _create_checkpoints(checkpoint_root: Path) -> None:
    """Create the minimum checkpoint layout consumed by the launcher."""
    tokenizer = "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
    for model, revision in MODEL_REVISIONS.items():
        checkpoint = checkpoint_root / model
        weights = checkpoint / "release" / "mp_rank_00" / "model_optim_rng.pt"
        weights.parent.mkdir(parents=True)
        weights.touch()
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("release\n")
        (checkpoint / tokenizer).touch()
        tree = checkpoint / ".cache" / "huggingface" / "trees" / f"{revision}.json"
        tree.parent.mkdir(parents=True)
        tree.write_text("{}\n")


def _write_artifact_writer(path: Path) -> None:
    """Write the deterministic artifact producer used by launcher tests."""
    path.write_text(
        '''"""Produce complete or intentionally invalid Phase 3 artifacts."""

import json
import os
import sys
from pathlib import Path

MODEL_REVISIONS = {
    "mamba2-8b": "b915550c63ba9359f88f44d1f6a600d85af27302",
    "mamba2-hybrid-8b": "35e8852e2240b350ac2fe2a3b8aa341b5930018e",
}


def write_kv(output: Path, model: str, revision: str, bad_output: str | None) -> None:
    """Write one 50 by 10 key-value control artifact."""
    with output.open("w") as stream:
        for index in range(500):
            control_id, position = divmod(index, 10)
            if bad_output == "kv-duplicate" and index == 499:
                control_id, position = 0, 0
            record = {
                "control_id": control_id,
                "condition": f"kv_position_{position}",
                "score": 1.0,
                "model_revision": revision,
                "model_key": model,
                "execution_path": "megatron_cuda_kernels",
            }
            stream.write(json.dumps(record) + "\\n")
    if bad_output == "kv-malformed":
        output.write_text("{\\n")


def write_sweep(output: Path, model: str, revision: str, bad_output: str | None) -> None:
    """Write one 800 by 12 position-sweep artifact."""
    count = 9599 if bad_output == "sweep-incomplete" else 9600
    with output.open("w") as stream:
        for index in range(count):
            question, slot = divmod(index, 12)
            condition = "closed_book" if slot == 0 else "oracle" if slot == 1 else "gold"
            position = None if slot < 2 else slot - 2
            if bad_output == "sweep-null" and index == 10:
                score = None
            elif bad_output == "sweep-nan" and index == 10:
                score = float("nan")
            else:
                score = 1.0
            record = {
                "question_id": f"q{question}",
                "condition": condition,
                "gold_position": position,
                "score": score,
                "score_normalized_em": score,
                "score_first_line": score,
                "floor_accuracy": 0.0,
                "ceiling_accuracy": 1.0,
                "model_revision": revision,
                "software_versions": {
                    "model_key": model,
                    "execution_path": "megatron_cuda_kernels",
                },
            }
            stream.write(json.dumps(record) + "\\n")
    if bad_output == "sweep-failures":
        output.with_suffix(".failures.jsonl").write_text('{"error": "boom"}\\n')


def main(arguments: list[str]) -> None:
    """Write the artifact selected by the fixture launcher."""
    model, stage = arguments
    revision = MODEL_REVISIONS[model]
    bad_output = os.environ.get("BAD_OUTPUT") if model == "mamba2-8b" else None
    output_directory = Path(os.environ["OUT_DIR"])
    if stage == "kv":
        write_kv(output_directory / f"{model}-positive-control.jsonl", model, revision, bad_output)
    else:
        write_sweep(output_directory / f"{model}-sweep.jsonl", model, revision, bad_output)


main(sys.argv[1:])
'''
    )


def _write_fake_commands(fake_bin: Path) -> None:
    """Provide deterministic git, hashing, and GPU inspection commands."""
    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" != -C ]; then
  exec "$REAL_GIT" "$@"
fi
repository=$2
command=$3
if [ "$command" = status ]; then
  if [ "$repository" = "$MIXING" ]; then
    printf '%s\\n' "${MIXING_GIT_STATUS:-}"
  else
    printf '%s\\n' "${MEGATRON_GIT_STATUS:-}"
  fi
elif [ "$repository" = "$MIXING" ]; then
  printf '%s\\n' "$MIXING_GIT_SHA"
else
  printf '%s\\n' "$MEGATRON_GIT_SHA"
fi
""",
    )
    _write_executable(
        fake_bin / "sha256sum",
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
*mamba2-hybrid-8b/release/mp_rank_00/model_optim_rng.pt)
  digest=$FAKE_HYBRID_WEIGHT_SHA256 ;;
*mamba2-8b/release/mp_rank_00/model_optim_rng.pt)
  digest=$FAKE_PURE_WEIGHT_SHA256 ;;
*/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model)
  digest=$FAKE_TOKENIZER_SHA256 ;;
*/latest_checkpointed_iteration.txt)
  digest=$FAKE_LATEST_SHA256 ;;
*/scripts/run_phase3_baremetal.sh)
  digest=$FAKE_LAUNCHER_SHA256 ;;
esac
if [ "$1" = "$PIQA" ]; then
  digest=$PIQA_SHA256
elif [ "$1" = "$DATA" ]; then
  digest=$DATA_SHA256
fi
printf '%s  %s\\n' "$digest" "$1"
""",
    )
    _write_executable(
        fake_bin / "nvidia-smi",
        """#!/usr/bin/env bash
printf 'NVIDIA-SMI test fixture\\n'
""",
    )


def _write_executable(path: Path, content: str) -> None:
    """Write one executable fixture script."""
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
