import json

import pytest

from app.services.result_saver import append_result


@pytest.mark.asyncio
async def test_append_creates_file_with_entry(tmp_path):
    path = tmp_path / "out" / "results.json"
    await append_result({"request_id": 1, "categories": [], "extracted_documents": []}, str(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == [{"request_id": 1, "categories": [], "extracted_documents": []}]


@pytest.mark.asyncio
async def test_append_accumulates_multiple_results(tmp_path):
    path = tmp_path / "results.json"
    await append_result({"request_id": 1}, str(path))
    await append_result({"request_id": 2}, str(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert [r["request_id"] for r in data] == [1, 2]


@pytest.mark.asyncio
async def test_blank_path_disables_saving(tmp_path):
    # Should be a no-op and not raise.
    await append_result({"request_id": 1}, "")


@pytest.mark.asyncio
async def test_corrupt_file_is_replaced(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("not valid json", encoding="utf-8")

    await append_result({"request_id": 9}, str(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == [{"request_id": 9}]
