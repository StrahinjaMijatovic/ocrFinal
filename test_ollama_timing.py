import asyncio
import time
from app.prompts.registry import registry
from app.pipeline.ocr_engine import extract_text_from_file
from app.services.ollama_client import OllamaClient
from pathlib import Path


async def test():
    test_dir = Path(r'C:\Users\strahinja\Desktop\Test\109537')
    client = OllamaClient()

    for f in sorted(test_dir.iterdir()):
        if f.suffix.lower() not in {'.pdf', '.jpg', '.jpeg', '.png'}:
            continue
        try:
            text, method = extract_text_from_file(f)
            if not text.strip():
                print(f'{f.name}: SKIPPED (no text, method={method})')
                continue
            prompt, temp = registry.format('structuring', 'v2_few_shot', ocr_text=text)
            print(f'{f.name}: prompt={len(prompt)} chars, sending...')
            start = time.time()
            result = await client.generate(prompt, temperature=temp)
            elapsed = time.time() - start
            dtype = result.get("document_type", "?")
            conf = result.get("confidence", "?")
            print(f'  -> {dtype} (conf={conf}) in {elapsed:.1f}s')
        except Exception as e:
            elapsed = time.time() - start if 'start' in dir() else 0
            print(f'  -> FAILED after {elapsed:.1f}s: {e}')

    await client.aclose()


asyncio.run(test())
