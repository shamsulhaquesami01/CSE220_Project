"""
imaging.py

The only place in the project that talks to Pillow and to the filesystem.

Keeping all I/O here means `dsp_utils` stays a pure NumPy module with no
third-party dependencies, so the algorithms the course is actually grading can
be imported and tested without Django or Pillow present.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
from django.conf import settings
from PIL import Image

from . import dsp_utils as dsp

# Formats we accept on upload. Anything Pillow can decode would work, but a
# short allowlist avoids surprises with exotic or animated formats.
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# Return the directory uploads are written to, creating it if needed.
def upload_dir() -> Path:
    path = Path(settings.MEDIA_ROOT) / settings.UPLOAD_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


# Return the directory generated panels are written to, creating it if needed.
def result_dir() -> Path:
    path = Path(settings.MEDIA_ROOT) / settings.RESULT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


# Return the media URL for a file living under MEDIA_ROOT.
def media_url(relative_path: str) -> str:
    return f"{settings.MEDIA_URL}{relative_path}".replace("\\", "/")


# Downscale an image so its longest edge is at most max_dim, preserving aspect.
def _fit_within(image: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image

    scale = max_dim / float(longest)
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    # Anti-aliased on purpose: the ingest downscale is itself a decimation, and
    # letting it alias would corrupt every experiment run afterwards.
    return dsp.resize(image, out_h, out_w, method="bilinear", antialias=True)


# Save an uploaded file to MEDIA_ROOT and return its identifier and geometry.
def store_upload(uploaded_file, grayscale: bool = False) -> dict:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    with Image.open(uploaded_file) as handle:
        handle.draft(None, None)
        pil_image = handle.convert("L" if grayscale else "RGB")
        array = np.asarray(pil_image)

    image = dsp.to_float(array)
    image = _fit_within(image, settings.IMAGE_LAB_MAX_DIM)

    # Store the ingest-normalised image as PNG so every later stage reads back
    # exactly the pixels the DSP code worked on, with no JPEG re-compression.
    image_id = uuid.uuid4().hex
    filename = f"{image_id}.png"
    save_array(image, upload_dir() / filename)

    height, width = image.shape[:2]
    return {
        "image_id": image_id,
        "url": media_url(f"{settings.UPLOAD_SUBDIR}/{filename}"),
        "width": int(width),
        "height": int(height),
        "channels": 1 if image.ndim == 2 else int(image.shape[2]),
    }


# Load a previously uploaded image by id as a float array in [0, 1].
def load_upload(image_id: str) -> np.ndarray:
    if not image_id or not image_id.isalnum():
        raise ValueError("Malformed image id")

    path = upload_dir() / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError("That image is no longer on the server. Please re-upload.")

    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB" if handle.mode != "L" else "L"))
    return dsp.to_float(array)


# Write a float array in [0, 1] to disk as a PNG.
def save_array(image: np.ndarray, path: Path) -> Path:
    data = dsp.to_uint8(image)
    mode = "L" if data.ndim == 2 else "RGB"
    Image.fromarray(data, mode=mode).save(path, format="PNG", optimize=True)
    return path


# Return a short deterministic digest of the request that produced a panel.
def panel_digest(image_id: str, op_id: str, params: dict, panel_key: str) -> str:
    payload = json.dumps(
        {"image": image_id, "op": op_id, "params": params, "panel": panel_key},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# Persist a result panel, reusing the file when the same request repeats.
def save_panel(image: np.ndarray, image_id: str, op_id: str, params: dict, panel_key: str) -> str:
    digest = panel_digest(image_id, op_id, params, panel_key)
    filename = f"{image_id}_{op_id}_{panel_key}_{digest}.png"
    path = result_dir() / filename

    # Content-addressed by request, so an identical re-run is a cache hit. The
    # live preview fires on every slider drag, and this keeps that cheap.
    if not path.exists():
        save_array(image, path)

    return media_url(f"{settings.RESULT_SUBDIR}/{filename}")


# Delete generated panels beyond the newest `keep` files.
def prune_results(keep: int = 400) -> int:
    files = sorted(result_dir().glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for stale in files[keep:]:
        try:
            stale.unlink()
            removed += 1
        except OSError:
            pass
    return removed
