import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
assert SPEC and SPEC.loader
summarize_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize_results)


def test_load_run_accepts_directory(tmp_path: Path) -> None:
    metadata = {
        "status": "completed",
        "processing_duration_seconds": 12.3456,
        "frames": {
            "extracted": 10,
            "selected": 8,
            "registered": 7,
            "registration_rate": 0.875,
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    run = summarize_results.load_run(tmp_path)
    table = summarize_results.summarize([run])

    assert f"| {tmp_path.name} | completed | no | 10 | 8 | 7 | 87.50% | 12.346 |" in table


def test_load_run_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text("not json", encoding="utf-8")

    try:
        summarize_results.load_run(path)
    except ValueError as exc:
        assert "Invalid JSON metadata" in str(exc)
    else:
        raise AssertionError("Expected invalid metadata to fail")


def test_main_returns_two_for_missing_metadata(tmp_path: Path, capsys) -> None:
    result = summarize_results.main([str(tmp_path)])

    assert result == 2
    assert "Could not read metadata" in capsys.readouterr().err
