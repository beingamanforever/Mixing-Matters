from mixing_matters.sink_scan import sink_mass_summary


def _record(model, qid, position, layer, mass):
    return {
        "model_key": model,
        "question_id": qid,
        "gold_position": position,
        "layer": layer,
        "sink_mass": mass,
    }


def test_sink_mass_summary_averages_per_model_position_layer():
    records = []
    # Two questions, ten positions, three layers, two models.
    for q in range(2):
        for position in range(10):
            for layer in range(3):
                records.append(
                    _record("pythia-2.8b", f"q{q}", position, layer, mass=0.4 + 0.1 * layer)
                )
                records.append(
                    _record("nemotron-h-8b", f"q{q}", position, layer, mass=0.2 + 0.05 * layer)
                )
    summary = sink_mass_summary(records)
    assert set(summary["models"]) == {"pythia-2.8b", "nemotron-h-8b"}
    for model in summary["models"]:
        entry = summary["by_model"][model]
        assert entry["layers"] == [0, 1, 2]
        assert entry["question_count"] == 2
        for position in range(10):
            per_layer = entry["positions"][position]["mean_sink_mass_per_layer"]
            assert len(per_layer) == 3
            expected_base = 0.4 if model == "pythia-2.8b" else 0.2
            step = 0.1 if model == "pythia-2.8b" else 0.05
            for layer, value in enumerate(per_layer):
                assert abs(value - (expected_base + step * layer)) < 1e-9
