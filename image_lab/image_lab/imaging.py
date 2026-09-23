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

_BLOB_ENDPOINT = "https://vercel.com/api/blob"
_BLOB_ID_PREFIX = "blob_"


def _blob_auth() -> tuple[str | None, str | None]:
    """Return (bearer_token, bare_store_id) for Vercel Blob.

    Modern Vercel Blob connections use the deployment's rotating OIDC token
    together with BLOB_STORE_ID. Older/manual setups may still expose a
    BLOB_READ_WRITE_TOKEN, so both modes are supported.
    """
    oidc = os.environ.get("VERCEL_OIDC_TOKEN")
    store_id = os.environ.get("BLOB_STORE_ID")

    if oidc and store_id:
        bare_store_id = store_id.removeprefix("store_")
        return oidc, bare_store_id

    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token:
        for key, value in os.environ.items():
            if key.endswith("BLOB_READ_WRITE_TOKEN") and value:
                token = value
                break

    if token:
        # Read/write tokens have the form vercel_blob_rw_<storeId>_<secret>.
        parts = token.split("_")
        parsed_store = parts[3] if len(parts) > 4 else None
        return token, parsed_store

    return None, None


def using_blob() -> bool:
    token, store_id = _blob_auth()
    return bool(token and store_id)


def _ensure_storage_ready() -> None:
    if os.environ.get("VERCEL") and not using_blob():
        raise ValueError(
            "Image storage is not configured for this deployment. "
            "Connect the Vercel Blob store to this project and make sure "
            "System Environment Variables are enabled."
        )


def _blob_put(pathname: str, payload: bytes, content_type: str = "image/png") -> str:
    token, store_id = _blob_auth()
    if not token or not store_id:
        raise RuntimeError("Vercel Blob credentials are not available.")

    # Match the current @vercel/blob server protocol: requests go through
    # Vercel's Blob control API, while the target store is supplied separately
    # because an OIDC token itself is project-scoped rather than store-scoped.
    request = Request(
        f"{_BLOB_ENDPOINT}?pathname={quote(pathname, safe='/')}",
        data=payload,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
            "x-vercel-blob-store-id": store_id,
            "x-api-version": "12",
            "x-vercel-blob-access": "public",
            "x-content-type": content_type,
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



