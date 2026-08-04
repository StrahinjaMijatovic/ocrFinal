import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.ollama_client import OllamaClient


@pytest.mark.asyncio
async def test_generate_returns_parsed_json():
    client = OllamaClient()
    mock_response = {"response": json.dumps({"document_type": "PASSPORT", "confidence": 0.9})}

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_post.return_value = mock_resp

        result = await client.generate("test prompt")
        assert result["document_type"] == "PASSPORT"
    await client.aclose()


@pytest.mark.asyncio
async def test_generate_retries_on_invalid_json():
    client = OllamaClient()
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if call_count < 2:
            resp.json.return_value = {"response": "not json at all"}
        else:
            resp.json.return_value = {"response": json.dumps({"ok": True})}
        return resp

    with patch.object(client._client, "post", side_effect=mock_post):
        result = await client.generate("test prompt", max_retries=3)
        assert result["ok"] is True
        assert call_count == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_health_check_true():
    client = OllamaClient()
    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        assert await client.health_check() is True
    await client.aclose()
