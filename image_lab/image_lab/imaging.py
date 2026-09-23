"""
imaging.py

Image/file I/O for Image Lab.

Local development uses MEDIA_ROOT just like before. On Vercel, if a Vercel Blob
store is connected, uploads and generated result PNGs are stored in Blob so
they survive across serverless requests and remain downloadable.

The DSP core remains pure NumPy and is intentionally unaware of storage.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
from django.conf import settings
from PIL import Image

from . import dsp_utils as dsp


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

_BLOB_ENDPOINT = "https://blob.vercel-storage.com"
_BLOB_ID_PREFIX = "blob_"


def _blob_token() -> str | None:
    # Standard Vercel Blob connection name.
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if token:
        return token

    # If the Blob resource was connected with a custom environment-variable
    # prefix, Vercel keeps the BLOB_READ_WRITE_TOKEN suffix. Accept that form
    # too so the deployment does not depend on the dashboard prefix choice.
    for key, value in os.environ.items():
        if key.endswith("BLOB_READ_WRITE_TOKEN") and value:
            return value

    return None


def using_blob() -> bool:
    return bool(_blob_token())


def _ensure_storage_ready() -> None:
    if os.environ.get("VERCEL") and not using_blob():
        raise ValueError(
            "Image storage is not configured for this deployment. "
            "Connect a Vercel Blob store to the project and redeploy."
        )


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------


def upload_dir() -> Path:
    path = Path(settings.MEDIA_ROOT) / settings.UPLOAD_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_dir() -> Path:
    path = Path(settings.MEDIA_ROOT) / settings.RESULT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def media_url(relative_path: str) -> str:
    return f"{settings.MEDIA_URL}{relative_path}".replace("\\", "/")


# ---------------------------------------------------------------------------
# Shared image encode/decode helpers
# ---------------------------------------------------------------------------


def _fit_within(image: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image

    scale = max_dim / float(longest)
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    return dsp.resize(image, out_h, out_w, method="bilinear", antialias=True)


def _png_bytes(image: np.ndarray) -> bytes:
    data = dsp.to_uint8(image)
    mode = "L" if data.ndim == 2 else "RGB"
    buffer = io.BytesIO()
    Image.fromarray(data, mode=mode).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _decode_image_bytes(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as handle:
        mode = "RGB" if handle.mode != "L" else "L"
        array = np.asarray(handle.convert(mode))
    return dsp.to_float(array)


# ---------------------------------------------------------------------------
# Vercel Blob backend
# ---------------------------------------------------------------------------


def _blob_put(pathname: str, payload: bytes, content_type: str = "image/png") -> str:
    token = _blob_token()
    if not token:
        raise RuntimeError("Vercel Blob is not configured.")

    # Vercel Blob's server upload endpoint accepts the pathname in the URL and
    # authenticates with the read/write token injected when the store is linked.
    request = Request(
        f"{_BLOB_ENDPOINT}/{quote(pathname, safe='/')}",
        data=payload,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
            "x-api-version": "7",
            "x-add-random-suffix": "1",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Vercel Blob upload failed ({exc.code}): {detail[:300]}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Vercel Blob: {exc.reason}") from exc

    try:
        result = json.loads(raw)
        return result["url"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("Vercel Blob returned an invalid response.") from exc


def _blob_get(url: str) -> bytes:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".blob.vercel-storage.com")
    ):
        raise ValueError("Malformed Blob URL")

    request = Request(url)
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(
                "That image is no longer available. Please re-upload."
            ) from exc
        raise RuntimeError(f"Vercel Blob read failed ({exc.code}).") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Vercel Blob: {exc.reason}") from exc


def _encode_blob_id(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii")
    return _BLOB_ID_PREFIX + encoded.rstrip("=")


def _decode_blob_id(image_id: str) -> str:
    if not image_id.startswith(_BLOB_ID_PREFIX):
        raise ValueError("Malformed image id")

    encoded = image_id[len(_BLOB_ID_PREFIX):]
    encoded += "=" * (-len(encoded) % 4)
    try:
        url = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise ValueError("Malformed image id") from exc

    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".blob.vercel-storage.com")
    ):
        raise ValueError("Malformed image id")
    return url


# ---------------------------------------------------------------------------
# Public storage API used by views.py
# ---------------------------------------------------------------------------


def store_upload(uploaded_file, grayscale: bool = False) -> dict:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    _ensure_storage_ready()

    with Image.open(uploaded_file) as handle:
        handle.draft(None, None)
        pil_image = handle.convert("L" if grayscale else "RGB")
        array = np.asarray(pil_image)

    image = dsp.to_float(array)
    image = _fit_within(image, settings.IMAGE_LAB_MAX_DIM)

    image_id = uuid.uuid4().hex
    filename = f"{image_id}.png"

    if using_blob():
        blob_url = _blob_put(
            f"{settings.UPLOAD_SUBDIR}/{filename}",
            _png_bytes(image),
        )
        returned_id = _encode_blob_id(blob_url)
        image_url = blob_url
    else:
        save_array(image, upload_dir() / filename)
        returned_id = image_id
        image_url = media_url(f"{settings.UPLOAD_SUBDIR}/{filename}")

    height, width = image.shape[:2]
    return {
        "image_id": returned_id,
        "url": image_url,
        "width": int(width),
        "height": int(height),
        "channels": 1 if image.ndim == 2 else int(image.shape[2]),
    }


def load_upload(image_id: str) -> np.ndarray:
    if image_id.startswith(_BLOB_ID_PREFIX):
        return _decode_image_bytes(_blob_get(_decode_blob_id(image_id)))

    if not image_id or not image_id.isalnum():
        raise ValueError("Malformed image id")

    path = upload_dir() / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError(
            "That image is no longer on the server. Please re-upload."
        )

    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB" if handle.mode != "L" else "L"))
    return dsp.to_float(array)


def save_array(image: np.ndarray, path: Path) -> Path:
    data = dsp.to_uint8(image)
    mode = "L" if data.ndim == 2 else "RGB"
    Image.fromarray(data, mode=mode).save(path, format="PNG", optimize=True)
    return path


def panel_digest(image_id: str, op_id: str, params: dict, panel_key: str) -> str:
    payload = json.dumps(
        {"image": image_id, "op": op_id, "params": params, "panel": panel_key},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def save_panel(
    image: np.ndarray,
    image_id: str,
    op_id: str,
    params: dict,
    panel_key: str,
) -> str:
    digest = panel_digest(image_id, op_id, params, panel_key)

    if using_blob():
        # Hash the image id rather than putting the encoded Blob URL in the
        # pathname. This keeps result paths short and opaque.
        source_key = hashlib.sha1(image_id.encode("utf-8")).hexdigest()[:12]
        filename = f"{source_key}_{op_id}_{panel_key}_{digest}.png"
        return _blob_put(
            f"{settings.RESULT_SUBDIR}/{filename}",
            _png_bytes(image),
        )

    filename = f"{image_id}_{op_id}_{panel_key}_{digest}.png"
    path = result_dir() / filename
    if not path.exists():
        save_array(image, path)
    return media_url(f"{settings.RESULT_SUBDIR}/{filename}")


def prune_results(keep: int = 400) -> int:
    # Blob objects are not pruned here because they may live on another
    # serverless instance. Local development keeps the previous bounded cache.
    if using_blob():
        return 0

    files = sorted(
        result_dir().glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for stale in files[keep:]:
        try:
            stale.unlink()
            removed += 1
        except OSError:
            pass
    return removed
