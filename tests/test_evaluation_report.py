
import json
from pathlib import Path

import pytest

import app.evaluation.runner as runner_module
from app.evaluation.runner import ResearchEvaluationRunner


@pytest.fixture
def evaluation_runner():
    return ResearchEvaluationRunner()


def test_save_report_creates_json_file(
    evaluation_runner,
    tmp_path,
    monkeypatch
):
    # Redirect report output to a temporary test directory
    monkeypatch.setattr(
        runner_module,
        "__file__",
        str(tmp_path / "runner.py")
    )

    report = {
        "total_evaluated_cases": 5,
        "average_precision": 1.0,
        "average_ragas_f1_score": 1.0
    }

    file_path = evaluation_runner.save_report(report)

    # Verify that the file exists
    assert Path(file_path).exists()

    # Verify JSON content
    with open(file_path, "r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report


def test_save_report_creates_reports_directory(
    evaluation_runner,
    tmp_path,
    monkeypatch
):
    monkeypatch.setattr(
        runner_module,
        "__file__",
        str(tmp_path / "runner.py")
    )

    report = {"total_evaluated_cases": 5}

    evaluation_runner.save_report(report)

    reports_dir = tmp_path / "reports"

    assert reports_dir.exists()
    assert reports_dir.is_dir()


def test_save_report_rejects_invalid_input(
    evaluation_runner
):
    with pytest.raises(TypeError):
        evaluation_runner.save_report("invalid report")


def test_save_report_generates_unique_filenames(
    evaluation_runner,
    tmp_path,
    monkeypatch
):
    monkeypatch.setattr(
        runner_module,
        "__file__",
        str(tmp_path / "runner.py")
    )

    report = {"total_evaluated_cases": 5}

    first_path = evaluation_runner.save_report(report)
    second_path = evaluation_runner.save_report(report)

    assert first_path != second_path
