"""
views.py

Three endpoints:

  GET  /              -- render the single-page GUI
  POST /api/upload/   -- accept a multipart image, normalise it, return its id
  POST /api/process/  -- run one registered operation, return panel URLs + metrics

The processing endpoint is a JSON API rather than a form post so the frontend
can re-run an operation on every slider drag and swap the images in place,
which is what makes the kernel editor feel live.
"""

from __future__ import annotations

import json
import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from . import dsp_utils as dsp
from . import imaging
from .operations import CLEAN_FILTERS, NOISE_MODELS, OPERATIONS, get_operation

logger = logging.getLogger(__name__)


# Render the single-page interface.
@require_GET
def index(request):
    context = {
        "operations": list(OPERATIONS.values()),
        "kernel_presets": json.dumps(dsp.KERNEL_PRESETS),
        "preset_names": list(dsp.KERNEL_PRESETS),
        "pad_modes": list(dsp.PAD_MODES),
        "noise_models": NOISE_MODELS,
        "clean_filters": CLEAN_FILTERS,
        "max_dim": settings.IMAGE_LAB_MAX_DIM,
        "max_upload_mb": settings.IMAGE_LAB_MAX_UPLOAD_BYTES // (1024 * 1024),
    }
    return render(request, "image_lab/index.html", context)


# Accept an uploaded image, normalise it, and return a handle to it.
@require_POST
def upload(request):
    uploaded = request.FILES.get("image")
    if uploaded is None:
        return JsonResponse({"error": "No file was submitted."}, status=400)

    if uploaded.size > settings.IMAGE_LAB_MAX_UPLOAD_BYTES:
        limit_mb = settings.IMAGE_LAB_MAX_UPLOAD_BYTES // (1024 * 1024)
        return JsonResponse({"error": f"File is larger than {limit_mb} MB."}, status=400)

    grayscale = request.POST.get("grayscale", "") in {"1", "true", "on"}

    try:
        info = imaging.store_upload(uploaded, grayscale=grayscale)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        logger.exception("Upload failed")
        return JsonResponse({"error": "Could not decode that image."}, status=400)

    # Opportunistic housekeeping so the results cache cannot grow without bound.
    imaging.prune_results()
    return JsonResponse(info)


# Run one registered operation against a stored image and return the results.
@require_POST
def process(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Request body must be JSON."}, status=400)

    image_id = str(payload.get("image_id", ""))
    op_id = str(payload.get("op", ""))
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        return JsonResponse({"error": "`params` must be an object."}, status=400)

    try:
        operation = get_operation(op_id)
    except KeyError:
        return JsonResponse({"error": f"Unknown operation '{op_id}'."}, status=400)

    try:
        image = imaging.load_upload(image_id)
    except (ValueError, FileNotFoundError) as exc:
        return JsonResponse({"error": str(exc)}, status=404)

    try:
        result = operation.handler(image, params)
    except ValueError as exc:
        # Parameter-level problems are the user's to fix, so surface the text.
        return JsonResponse({"error": str(exc)}, status=400)
    except MemoryError:
        return JsonResponse(
            {"error": "Ran out of memory. Try a smaller kernel or image."}, status=400
        )
    except Exception:
        logger.exception("Operation %s failed", op_id)
        return JsonResponse({"error": "The operation failed. Check the server log."}, status=500)

    panels = [
        {
            "key": panel.key,
            "label": panel.label,
            "caption": panel.caption,
            "url": imaging.save_panel(panel.image, image_id, op_id, params, panel.key),
            "width": int(panel.image.shape[1]),
            "height": int(panel.image.shape[0]),
        }
        for panel in result.panels
    ]

    return JsonResponse(
        {
            "op": op_id,
            "panels": panels,
            "metrics": [
                {"label": m.label, "value": m.value, "hint": m.hint} for m in result.metrics
            ],
            "notes": result.notes,
        }
    )
