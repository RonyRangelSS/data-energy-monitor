import pytest

from energy_monitor.experiments import run_record


def fake_git(status: str | None):
    calls: list[tuple[str, ...]] = []

    def run(*arguments: str) -> str | None:
        calls.append(arguments)
        return status if arguments[0] == "status" else "abc123"

    return run, calls


def test_generated_reports_do_not_count_as_code_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    run, calls = fake_git("")
    monkeypatch.setattr(run_record, "run_git", run)

    version = run_record.describe_code_version()

    assert version == {"commit": "abc123", "has_uncommitted_changes": False}
    assert run_record.GENERATED_REPORTS_PATHSPEC in calls[0]


def test_code_changes_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = fake_git(" M src/file.py")
    monkeypatch.setattr(run_record, "run_git", run)

    assert run_record.describe_code_version()["has_uncommitted_changes"] is True


def test_unknown_status_stays_unknown_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = fake_git(None)
    monkeypatch.setattr(run_record, "run_git", run)

    assert run_record.describe_code_version()["has_uncommitted_changes"] is None
