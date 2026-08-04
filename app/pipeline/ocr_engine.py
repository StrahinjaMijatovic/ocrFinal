import re
from pathlib import Path

import fitz  # pymupdf
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from app.config import settings

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# One PaddleOCR instance per recognition model, created lazily and cached.
_ocr_instances: dict[str, PaddleOCR] = {}

# Detects a Machine-Readable Zone line: a long run of the MRZ alphabet containing
# the "<<" filler that never occurs in normal prose. Used only to recognise identity
# documents so they can be skipped — their data comes from the database. We do NOT
# parse any fields out of the MRZ.
_MRZ_LINE = re.compile(r"[A-Z0-9<]{25,}")

# ICAO 9303 encodes the document kind in the first MRZ character:
# P = passport, I/A/C = ID card / residence permit, V = visa, ...
# We skip only passports and ID cards; visas (V) and other docs are kept.
_SKIP_MRZ_TYPES = ("P", "I")


def _ocr_langs() -> list[str]:
    return [lang.strip() for lang in settings.ocr_languages.split(",") if lang.strip()]


def get_ocr(lang: str) -> PaddleOCR:
    if lang not in _ocr_instances:
        _ocr_instances[lang] = PaddleOCR(
            use_angle_cls=True, lang=lang, use_gpu=False, show_log=False
        )
    return _ocr_instances[lang]


def is_photo_only(text: str) -> bool:
    """A file is treated as photo-only (e.g. an applicant portrait) when OCR finds
    no usable text in it. This is content-based, so the photo can have any filename."""
    return len(text.strip()) < 3


def is_identity_mrz(text: str) -> bool:
    """True when the text contains a PASSPORT or ID-CARD Machine-Readable Zone — the
    documents we skip (their data comes from the DB). Visas (MRZ type 'V') and other
    MRZ documents are NOT flagged. Language-independent, near-zero false positives.

    The first MRZ line encountered (top-to-bottom) is line 1, whose first character is
    the ICAO document-type code."""
    for line in text.splitlines():
        s = line.replace(" ", "").upper()
        if "<<" in s and _MRZ_LINE.fullmatch(s):
            return s[0] in _SKIP_MRZ_TYPES
    return False


def extract_pdf_text_layer(pdf_path: Path) -> str:
    """Return the embedded text layer of a digital PDF (no OCR). Empty string for
    scanned/image PDFs or on any read error."""
    try:
        with fitz.open(str(pdf_path)) as doc:
            return "\n".join(page.get_text().strip() for page in doc).strip()
    except Exception:
        return ""


def _ocr_image(image: np.ndarray) -> tuple[str, float]:
    """Run every configured OCR model on the image and return the (text, mean_confidence)
    of whichever model read it with the highest confidence. Language-agnostic."""
    best_text, best_conf = "", -1.0
    for lang in _ocr_langs():
        result = get_ocr(lang).ocr(image, cls=True)
        if not result or not result[0]:
            continue
        lines = [ln for ln in result[0] if ln and len(ln) >= 2]
        if not lines:
            continue
        text = "\n".join(ln[1][0] for ln in lines)
        conf = sum(ln[1][1] for ln in lines) / len(lines)
        if conf > best_conf:
            best_text, best_conf = text, conf
    return best_text, max(best_conf, 0.0)


def _pdf_to_images(pdf_path: Path) -> list[np.ndarray]:
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=200, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    return images


def render_to_images(file_path: Path) -> list[np.ndarray]:
    """Render a document to one RGB image array per page (PDFs) or a single image."""
    suffix = file_path.suffix.lower()
    if suffix in PHOTO_EXTENSIONS:
        return [np.array(Image.open(file_path).convert("RGB"))]
    if suffix == ".pdf":
        return _pdf_to_images(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def extract_text_from_file(file_path: Path) -> tuple[str, float]:
    """Return (ocr_text, mean_confidence). Photo-only files yield an empty string.
    The confidence (0.0-1.0) drives the optional vision-model fallback for poor scans."""
    results = [_ocr_image(img) for img in render_to_images(file_path)]
    texts = [t for t, _ in results]
    confs = [c for _, c in results if c > 0]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return "\n\n--- PAGE BREAK ---\n\n".join(texts), mean_conf
