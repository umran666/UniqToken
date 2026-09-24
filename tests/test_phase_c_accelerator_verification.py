"""Independent verification of the bounded Phase C accelerator probes."""

from pathlib import Path

import pytest

from tools import verify_phase_c_accelerator_calibrations as verify


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "artifacts" / "modal-runtime-gate" / "receipts"


def test_independent_verifier_accepts_frozen_probe_evidence():
    result = verify.verify(
        RECEIPTS / "phase-c-t4-calibration-measurement.json",
        RECEIPTS / "phase-c-t4-calibration-billing-review.json",
        RECEIPTS / "phase-c-a100-40gb-calibration-measurement.json",
        RECEIPTS / "phase-c-a100-40gb-calibration-billing-review.json",
        RECEIPTS / "phase-c-feasibility-parallel-reference.json",
        "cea54b22864823cf9aa054e410dee652451e5886",
    )
    assert [row["decision"] for row in result["records"]] == ["FAIL", "FAIL"]
    assert result["phase_c_conditions_started"] == 0
    assert result["validation_scored"] is False
    assert result["test_split_opened"] is False
    assert result["launch_authorized"] is False


def test_independent_verifier_rejects_tampered_content(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text('{"content_sha256":"bad","value":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        verify.read_immutable(path)


def test_independent_verifier_uses_exact_feasibility_file_hash(tmp_path):
    path = tmp_path / "feasibility.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="feasibility artifact changed"):
        verify.verify(None, None, None, None, path, "commit")
