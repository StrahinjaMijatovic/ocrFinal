---
name: ollama-cpu-integration
description: Use when integrating Ollama LLM calls in a CPU-only Python environment — especially with slow models (7B+, 20-60s per call), JSON output enforcement, httpx async client configuration, retry logic for parse failures, and health checks. Covers timeout tuning, RAM constraints, Windows-specific setup, and production patterns for offline inference pipelines.
---

# Ollama CPU Integration

Patterns for reliable Ollama integration on CPU-only hardware (no GPU, Windows, 7B models).

## Key Constraints for This Project

| Factor | Value |
|--------|-------|
| Model | Qwen2.5:7B-Instruct (Q4_K_M) |
| RAM usage | ~4.5 GB |
| Latency per call | 20-40s structuring, 30-60s categorization |
| Transport | httpx async, `http://localhost:11434` |
| JSON enforcement | Ollama `format: "json"` + retry on parse fail |

## httpx Client Configuration

```python
import httpx

class OllamaClient:
    def __init__(self) -> None:
        # Timeout must be >> worst-case inference time
        # 180s covers even slow CPU days; never use default 5s
        self._client = httpx.AsyncClient(timeout=180.0)
```

**Never** use `httpx.AsyncClient()` without explicit timeout — default is 5s and will silently fail on CPU inference.

## JSON Retry Pattern

Ollama's `format: "json"` reduces but does not eliminate non-JSON responses. Always retry:

```python
async def generate(self, prompt: str, temperature: float = 0.1, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        # On retry, append stronger enforcement message
        actual_prompt = (
            prompt if attempt == 0
            else prompt + "\n\nIMPORTANT: Return ONLY raw JSON. No markdown. No explanation."
        )
        response = await self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": actual_prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        response.raise_for_status()
        raw = response.json()["response"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise ValueError(f"Ollama returned invalid JSON after {max_retries} attempts: {raw[:200]}")
    # unreachable, satisfies type checker
    raise RuntimeError("generate loop exhausted")
```

## Health Check

Check Ollama is up AND model is loaded before starting pipeline — saves 5-7 min job from failing at step 4:

```python
async def health_check(self) -> bool:
    try:
        resp = await self._client.get(f"{self._base_url}/api/tags")
        if resp.status_code != 200:
            return False
        # Verify target model is actually available
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(self._model in m for m in models)
    except Exception:
        return False
```

## Temperature Settings

| Use case | Temperature |
|----------|-------------|
| Document structuring (JSON extraction) | `0.1` |
| Categorization (classification) | `0.1` |
| Never use >0.3 for structured output | — |

Low temperature is critical on CPU — it reduces variance which matters more at slow inference speed (you can't afford retries).

## RAM Considerations

- Qwen2.5:7B Q4_K_M = ~4.5 GB RAM
- Machine has 32 GB → safe
- Only ONE model loaded at a time (Ollama unloads on idle by default)
- If running multiple pipelines in parallel, add `OLLAMA_NUM_PARALLEL=1` to Ollama env

## Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `httpx.ReadTimeout` | timeout too short | Use `timeout=180.0` |
| `json.JSONDecodeError` on valid-looking text | Model added markdown wrapper | Retry with stronger instruction |
| `ConnectionRefusedError` on `localhost:11434` | Ollama not running | Health check before job start |
| Response has `"response": ""` | Model still loading | Add startup wait or retry |
| Very slow first call (~2x normal) | Model cold start | Expected; subsequent calls are faster |

## Windows-Specific Notes

- Ollama on Windows runs as a system tray app; start via `ollama serve` or the tray icon
- `OLLAMA_HOST=0.0.0.0` not needed for localhost-only use
- PaddleOCR and Ollama can run concurrently — OCR uses CPU cores, Ollama uses RAM; no conflict
- Do NOT run OCR and LLM call in true parallel on this hardware — RAM + CPU contention causes both to slow down; pipeline them sequentially

## Async Resource Cleanup

Always close the client when the pipeline is done:

```python
async def aclose(self) -> None:
    await self._client.aclose()

# In FastAPI lifespan or pipeline teardown:
# await ollama_client.aclose()
```
