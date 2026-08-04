import asyncio
import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from app.pipeline.ocr_engine import render_to_images
from app.services.ollama_client import OllamaClient

_TRANSCRIBE_PROMPT = (
    "Transcribe ALL text visible in this document image exactly as written, "
    "preserving line breaks and the original languages and scripts. "
    "Do not translate, summarise, or add commentary — output only the transcribed text."
)

# Cap the longest image side to keep the request payload and latency reasonable.
_MAX_SIDE = 1600


def _image_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    img.thumbnail((_MAX_SIDE, _MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def vision_transcribe(file_path: Path, ollama: OllamaClient) -> str:
    """Re-read a document with the vision model as a higher-quality OCR fallback for
    poor scans. Renders every page and returns the combined transcribed text."""
    images = await asyncio.to_thread(render_to_images, file_path)
    b64s = [_image_to_b64(img) for img in images]
    return await ollama.generate_vision(_TRANSCRIBE_PROMPT, b64s)
