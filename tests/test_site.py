import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mixing_matters.models import MODELS
from mixing_matters.site import CURVE_PANELS, LABELS, build_site_data, write_site_data

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
COMMITTED = WEB / "data" / "results.json"


@pytest.fixture(scope="module")
def data():
    return build_site_data(ROOT)


def test_labels_cover_the_model_registry():
    assert set(LABELS) == set(MODELS)


def test_panels_render_every_declared_model(data):
    assert [panel["id"] for panel in data["panels"]] == [key for key, _, _, _ in CURVE_PANELS]
    for panel in data["panels"]:
        for model in panel["models"]:
            assert len(panel["curves"][model]) == 10
            assert [point["position"] for point in panel["curves"][model]] == list(range(1, 11))
            assert set(panel["edges"][model]) == {"primacy", "recency"}
            assert set(panel["floor_ceiling"][model]) == {"floor_accuracy", "ceiling_accuracy"}


def test_edges_match_the_committed_phase2_summary(data):
    summary = json.loads((ROOT / "artifacts/phase2/report/phase2-summary.json").read_text())
    panel = next(item for item in data["panels"] if item["id"] == "phase2")
    for model, edges in summary["edges"].items():
        for edge in ("primacy", "recency"):
            point = panel["edges"][model][edge]
            assert point["estimate"] == edges[edge]["estimate"]
            assert point["ci"] == [edges[edge]["ci_low"], edges[edge]["ci_high"]]
            assert point["p"] == edges[edge]["p_value_holm"]


def test_phase3_and_phase5_edges_come_from_their_control_blocks(data):
    phase3 = json.loads((ROOT / "artifacts/phase3/report/phase3-summary.json").read_text())
    control = phase3["attention_control"]
    panel = next(item for item in data["panels"] if item["id"] == "phase3")
    assert (
        panel["edges"][control["hybrid_model"]]["primacy"]["estimate"]
        == (control["hybrid_edges"]["primacy"]["estimate"])
    )

    phase5 = json.loads((ROOT / "artifacts/phase5/report/phase5-summary.json").read_text())
    corpus = phase5["data_control"]
    panel = next(item for item in data["panels"] if item["id"] == "phase5")
    assert (
        panel["edges"][corpus["slimpajama_model"]]["recency"]["estimate"]
        == (corpus["slimpajama_edges"]["recency"]["estimate"])
    )


def test_contrasts_cover_every_paired_comparison(data):
    kinds = [row["kind"] for row in data["contrasts"]]
    assert kinds.count("architecture") == 3
    assert kinds.count("attention") == 1
    assert kinds.count("corpus") == 1
    assert kinds.count("scale") == 5
    assert kinds.count("production") == 3
    for row in data["contrasts"]:
        for edge in ("primacy", "recency"):
            assert row[edge]["ci"][0] <= row[edge]["estimate"] <= row[edge]["ci"][1]


def test_scale_trend_is_ordered_by_size(data):
    sizes = [entry["params_millions"] for entry in data["scale"]]
    assert sizes == sorted(sizes)
    assert len(data["scale"]) == 5


def test_committed_site_data_is_current(tmp_path):
    """The page ships a generated file; drift from artifacts/ is a bug."""
    fresh = write_site_data(ROOT, tmp_path / "results.json")
    assert COMMITTED.read_text() == fresh.read_text()


def test_page_data_builds_without_matplotlib(tmp_path):
    """The Pages workflow builds the page on a bare interpreter.

    Running the module through `cli` instead would import `figures` and fail on
    a runner that has not installed the project, so block matplotlib and prove
    the module entry point never reaches for it.
    """
    output = tmp_path / "results.json"
    script = f"""
import runpy, sys


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "matplotlib":
            raise ImportError("building page data must not require matplotlib")
        return None


sys.meta_path.insert(0, Block())
sys.argv = ["site", "--root", {str(ROOT)!r}, "--output", {str(output)!r}]
runpy.run_module("mixing_matters.site", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == COMMITTED.read_text()


def test_pages_workflow_runs_the_module_entry_point():
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "python -m mixing_matters.site --output web/data/results.json" in workflow


def test_page_only_references_files_that_exist():
    page = (WEB / "index.html").read_text()
    for path in re.findall(r'(?:href|src)="(?!https?:|#|data:)([^"]+)"', page):
        target = WEB / path
        # paper.pdf is copied in by the Pages workflow at deploy time.
        assert target.exists() or path == "paper.pdf", path


def test_nav_links_point_at_sections_that_exist():
    """Removing a section has to take its nav entry with it."""
    page = (WEB / "index.html").read_text()
    nav = page.split('<nav class="nav"', 1)[1].split("</nav>", 1)[0]
    anchors = re.findall(r'href="#([a-z-]+)"', nav)
    assert anchors
    for anchor in anchors:
        assert f'id="{anchor}"' in page, anchor


def test_page_mounts_every_section_the_script_fills():
    page = (WEB / "index.html").read_text()
    script = (WEB / "assets" / "app.js").read_text()
    assert 'src="assets/app.js"' in page
    assert 'href="assets/style.css"' in page
    for selector in set(re.findall(r'\$\("#([a-z-]+)"\)', script)):
        assert f'id="{selector}"' in page, selector
