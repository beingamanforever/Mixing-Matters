import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "paper" / "generate_figures.py"
SPEC = importlib.util.spec_from_file_location("paper_generate_figures", MODULE_PATH)
assert SPEC and SPEC.loader
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    output = tmp_path_factory.mktemp("paper-figures")
    paths = figures.generate_all(output)
    return output, paths


def test_generates_expected_vector_pairs(generated):
    _, paths = generated
    expected = {f"{stem}.{suffix}" for stem in figures.FIGURES for suffix in ("svg", "pdf")}
    assert {path.name for path in paths} == expected
    assert all(path.stat().st_size > 1_000 for path in paths)

    for path in paths:
        if path.suffix == ".svg":
            svg = path.read_text()
            assert svg.lstrip().startswith("<?xml")
            assert all(line == line.rstrip() for line in svg.splitlines())
            assert "<image " not in svg
            assert "base64" not in svg
            assert "Sources: artifacts/" in svg
            assert "Encoding: attention=blue" in svg
        else:
            assert path.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize(
    ("name", "labels", "sources"),
    [
        (
            "paper-phase2-position.svg",
            ("Pythia 2.8B", "Mamba 2.8B", "Mamba-2 2.7B"),
            ("artifacts/phase2/report/phase2-summary.json",),
        ),
        (
            "paper-phase3-position.svg",
            ("Mamba-2 8B", "Hybrid Mamba-2 8B"),
            ("artifacts/phase3/report/phase3-summary.json",),
        ),
        (
            "paper-phase4-scale.svg",
            ("Pythia minus Mamba", "Mean parameters (millions)"),
            ("artifacts/phase4/report/phase4-summary.json",),
        ),
        (
            "paper-phase5-corpus.svg",
            ("Pile", "SlimPajama"),
            ("artifacts/phase5/report/phase5-summary.json",),
        ),
        (
            "paper-phase1-calibration.svg",
            ("Key-value slot", "Calibration condition", "Floor", "Ceiling"),
            ("artifacts/phase1/figures/figures-summary.json",),
        ),
        (
            "paper-phase6-task.svg",
            ("QA", "Needle, 2048 tokens", "Primacy", "Recency", "Edge effect"),
            ("artifacts/phase6/report/phase6-summary.json",),
        ),
        (
            "paper-phase7-mechanisms.svg",
            ("Final-layer sink mass", "5-fold probe accuracy", "Prompt variant"),
            (
                "artifacts/phase7-mechanisms/report/phase7-summary.json",
                "artifacts/phase7-mechanisms/4d-probe/mamba-2.8b-layer32-probe.json",
            ),
        ),
        (
            "paper-phase8-production.svg",
            ("Nemotron-H 8B", "Llama 3.1 8B", "Qwen2.5 7B"),
            ("artifacts/phase8/report/phase8-summary.json",),
        ),
    ],
)
def test_svg_labels_and_sources_are_traceable(generated, name, labels, sources):
    output, _ = generated
    svg = (output / name).read_text()
    for label in (*labels, *sources):
        assert label in svg


def test_family_colors_and_markers_are_stable(generated):
    output, _ = generated
    phase2 = (output / "paper-phase2-position.svg").read_text().lower()
    assert figures.BLUE.lower() in phase2
    assert figures.ORANGE.lower() in phase2
    assert figures.STYLES["mamba-2.8b"] == (figures.ORANGE, "o")
    assert figures.STYLES["mamba-2.8b-slimpj"] == (figures.ORANGE, "^")
    assert figures.STYLES["mamba2-2.7b"] == (figures.ORANGE, "D")
    assert figures.STYLES["pythia-2.8b"] == (figures.BLUE, "*")
    assert "attention=blue" in phase2
    assert "state-space=orange" in phase2

    allowed = {
        figures.BLUE.lower(),
        figures.ORANGE.lower(),
        figures.GRAY.lower(),
        figures.BLACK.lower(),
        "#000000",
        "#777777",
        "#d7dadd",
        "#ffffff",
    }
    for path in output.glob("*.svg"):
        assert set(re.findall(r"#[0-9a-f]{6}", path.read_text().lower())) <= allowed


def test_phase7_uses_reported_final_layer_sink_mass():
    sink = figures._read(figures.SINK)["by_model"]
    assert figures._final_sink_mass(sink["pythia-160m"]) == pytest.approx(0.0030669, abs=1e-6)
    assert figures._final_sink_mass(sink["pythia-2.8b"]) == pytest.approx(0.455022, abs=1e-6)


def test_regeneration_is_byte_deterministic(generated):
    output, paths = generated
    first = {path.name: hashlib.sha256(path.read_bytes()).digest() for path in paths}
    regenerated = figures.generate_all(output)
    second = {path.name: hashlib.sha256(path.read_bytes()).digest() for path in regenerated}
    assert second == first


def test_refuses_to_replace_symlink(tmp_path):
    output = tmp_path / "figures"
    output.mkdir()
    victim = tmp_path / "victim.svg"
    victim.write_text("preserve me")
    (output / "paper-phase2-position.svg").symlink_to(victim)

    with pytest.raises(FileExistsError, match="non-regular output"):
        figures._phase2(output)

    assert victim.read_text() == "preserve me"
