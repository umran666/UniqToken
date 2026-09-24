from tools import phase_c_feasibility as gate
from pathlib import Path
import pytest


def test_analytical_schedule_counts_eos_and_windows():
    result = gate.analytical_schedule([[1, 2], list(range(128))])
    assert result["training_steps"] == 3
    assert result["training_target_tokens"] == 132
    assert result["training_sequence_length_squared_sum"] == 3 * 3 + 128 * 128 + 1
    assert result["actual_analytical_flops"] == (result["core_analytical_flops"] + result["output_projection_flops"])


def test_modal_cost_components_are_explicit():
    components, total = gate.hourly_cost()
    assert set(components) == {"l4_gpu", "four_cpu_cores", "sixteen_gib_memory"}
    assert total == sum(components.values())
    assert total > components["l4_gpu"]


@pytest.mark.parametrize("name", gate.phase_c.NAMES)
def test_process_encoding_matches_serial_frozen_artifacts(name):
    root = Path("artifacts/phase-a-screen-modal-f57b93d/screen-migrated-f57b93d")
    if not (root / "ledger.json").exists():
        pytest.skip("frozen tokenizer artifacts unavailable")
    source = next(
        row
        for row in gate.h.read_json(root / "ledger.json")["records"]
        if row["tokenizer"] == name and row["vocab_budget"] == 16384
    )
    texts = [
        "A short document.",
        "Longer text " * 100,
        "\u4e2d\u6587 \u0939\u093f\u0928\u094d\u0926\u0940",
        "def foo():\n    return 42\n",
    ]
    serial = gate.document_lengths(source, root, texts, 1)
    parallel = gate.document_lengths(source, root, texts, 2)
    assert parallel == serial
    assert gate.schedule_lengths(parallel) == gate.schedule_lengths(serial)


def test_worker_failure_propagates():
    with pytest.raises(Exception):
        gate.document_lengths({"artifact": "missing", "artifact_hashes": {}}, Path.cwd(), ["text"], 2)
