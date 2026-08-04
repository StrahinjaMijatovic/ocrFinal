import asyncio
import json
import httpx
from app.config import settings


class OllamaClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=600.0)
        self._model = settings.ollama_model
        self._base_url = settings.ollama_base_url

    async def generate(self, prompt: str, temperature: float = 0.1, max_retries: int = 3) -> dict:
        """Call Ollama /api/generate with JSON format enforcement.

        Retries on invalid JSON (re-prompting for raw JSON) as well as on transient
        network/HTTP errors (with exponential backoff). Raises after all attempts fail.
        """
        last_error: Exception | None = None
        for attempt in range(max_retries):
            actual_prompt = (
                prompt
                if attempt == 0
                else prompt + "\n\nIMPORTANT: Return ONLY raw JSON. No markdown. No explanation."
            )
            try:
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
                return json.loads(response.json()["response"])
            except json.JSONDecodeError as exc:
                # Bad model output — retry immediately with a stronger enforcement message.
                last_error = exc
            except (httpx.HTTPError, KeyError) as exc:
                # Transient network/server error — back off before retrying.
                last_error = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 10))

        raise ValueError(f"Ollama call failed after {max_retries} attempts: {last_error}")

    async def generate_vision(self, prompt: str, images: list[str], temperature: float = 0.1) -> str:
        """Call the configured vision model with base64 images, returning the raw text
        response (no JSON enforcement). Used as a higher-quality OCR fallback."""
        response = await self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": settings.vision_model,
                "prompt": prompt,
                "images": images,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        response.raise_for_status()
        return response.json()["response"]

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
