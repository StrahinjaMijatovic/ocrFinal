import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def isolate_results_file(tmp_path, monkeypatch):
    """Redirect the results-saver output to a temp file so tests never write to the
    real runs/ directory."""
    monkeypatch.setattr(
        "app.config.settings.results_file_path",
        str(tmp_path / "results.json"),
    )
