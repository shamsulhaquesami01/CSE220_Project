"""Rotation/scale/translation invariant image registration using Fourier-Mellin ideas.

The matcher is deliberately classical DSP: no OpenCV feature detector and no ML.
Translation is removed by using Fourier magnitude, rotation/scale become shifts in
log-polar coordinates, and phase correlation estimates those shifts.  After
rotation/scale correction a second phase-correlation stage estimates translation.
"""

from __future__ import annotations

import time

import numpy as np

from . import dsp_utils as dsp
from .operations import (
    Metric,
    OpResult,
    Panel,
    _as_bool,
    _as_choice,
    _as_float,
    _fmt,
    register,
)


WORK_MAX_SIDE = 768


def _rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.shape[-1] == 1:
        return np.repeat(image, 3, axis=2)
    return image[..., :3]


def _bilinear_sample(
    image: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape[:2]

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    valid = (
        (ys >= 0.0)
        & (ys <= h - 1)
        & (xs >= 0.0)
        & (xs <= w - 1)
    )
    wy = ys - y0
    wx = xs - x0

    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)

    if image.ndim == 2:
        out = (
            image[y0c, x0c] * (1.0 - wy) * (1.0 - wx)
            + image[y0c, x1c] * (1.0 - wy) * wx
            + image[y1c, x0c] * wy * (1.0 - wx)
            + image[y1c, x1c] * wy * wx
        )
        return np.where(valid, out, fill)

    wy3 = wy[..., None]
    wx3 = wx[..., None]
    out = (
        image[y0c, x0c] * (1.0 - wy3) * (1.0 - wx3)
        + image[y0c, x1c] * (1.0 - wy3) * wx3
        + image[y1c, x0c] * wy3 * (1.0 - wx3)
        + image[y1c, x1c] * wy3 * wx3
    )
    return np.where(valid[..., None], out, fill)


def warp_similarity(
    image: np.ndarray,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    out_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply a centre-based similarity transform by inverse-mapped bilinear sampling.

    Forward geometry is:
        p_out = scale * R(angle) * (p_in - centre_in) + centre_out + [tx, ty]
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape[:2]
    if scale <= 0:
        raise ValueError("Scale must be positive")

    out_h, out_w = out_shape or (h, w)
    cy, cx = (h - 1.0) / 2.0, (w - 1.0) / 2.0
    ocy, ocx = (out_h - 1.0) / 2.0, (out_w - 1.0) / 2.0

    yy, xx = np.mgrid[:out_h, :out_w]
    xd = xx - ocx - float(tx)
    yd = yy - ocy - float(ty)

    theta = np.deg2rad(float(angle_deg))
    cosine = np.cos(theta)
    sine = np.sin(theta)

    xs = (cosine * xd + sine * yd) / scale + cx
    ys = (-sine * xd + cosine * yd) / scale + cy
    return _bilinear_sample(image, ys, xs, fill=0.0)


def _hann2(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    wy = np.hanning(h) if h > 2 else np.ones(h)
    wx = np.hanning(w) if w > 2 else np.ones(w)
    return np.outer(wy, wx)


def _normalise_display(plane: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    plane = np.asarray(plane, dtype=np.float64)
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros_like(plane)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo + 1e-12:
        return np.zeros_like(plane)
    view = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    return view ** gamma


def _fft_magnitude(
    image: np.ndarray,
    use_hann: bool,
) -> np.ndarray:
    gray = np.asarray(dsp.to_gray(image), dtype=np.float64)
    gray = gray - float(gray.mean())
    if use_hann:
        gray = gray * _hann2(gray.shape)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(spectrum))


def log_polar_transform(
    spectrum: np.ndarray,
    angle_samples: int = 360,
    radial_samples: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a centred spectrum on (theta, log-radius) coordinates."""
    spectrum = np.asarray(spectrum, dtype=np.float64)
    h, w = spectrum.shape
    if min(h, w) < 40:
        raise ValueError("Images are too small for reliable Fourier-Mellin matching")

    cy, cx = (h - 1.0) / 2.0, (w - 1.0) / 2.0
    r_min = 2.0
    r_max = min(h, w) / 2.0 - 2.0
    radial_samples = int(max(64, min(radial_samples, max(64, int(r_max * 2.0)))))
    angle_samples = int(max(180, angle_samples))

    log_radii = np.linspace(np.log(r_min), np.log(r_max), radial_samples)
    radii = np.exp(log_radii)
    angles = np.linspace(0.0, 2.0 * np.pi, angle_samples, endpoint=False)

    theta = angles[:, None]
    radius = radii[None, :]
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)
    sampled = _bilinear_sample(spectrum, ys, xs, fill=0.0)
    return sampled, log_radii


def _parabolic_delta(left: float, centre: float, right: float) -> float:
    denominator = left - 2.0 * centre + right
    if abs(denominator) <= 1e-12:
        return 0.0
    delta = 0.5 * (left - right) / denominator
    return float(delta) if abs(delta) <= 1.0 else 0.0


def phase_correlation(
    reference: np.ndarray,
    query: np.ndarray,
    subpixel: bool = True,
) -> dict:
    """Return the cyclic shift to apply to query so it aligns with reference."""
    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if reference.shape != query.shape:
        raise ValueError("Phase-correlation arrays must have identical shapes")

    a = np.fft.fft2(reference)
    b = np.fft.fft2(query)
    cross = a * np.conj(b)
    magnitude = np.abs(cross)
    cross = cross / np.maximum(magnitude, 1e-12)
    correlation = np.real(np.fft.ifft2(cross))

    peak_index = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift = np.array(peak_index, dtype=np.float64)
    for axis, size in enumerate(correlation.shape):
        if shift[axis] > size // 2:
            shift[axis] -= size

    if subpixel:
        for axis, size in enumerate(correlation.shape):
            idx = list(peak_index)
            values = []
            for offset in (-1, 0, 1):
                sample = idx.copy()
                sample[axis] = (idx[axis] + offset) % size
                values.append(float(correlation[tuple(sample)]))
            shift[axis] += _parabolic_delta(*values)

    peak = float(correlation[peak_index])
    yy, xx = np.ogrid[: correlation.shape[0], : correlation.shape[1]]
    dy = np.minimum(
        np.abs(yy - peak_index[0]),
        correlation.shape[0] - np.abs(yy - peak_index[0]),
    )
    dx = np.minimum(
        np.abs(xx - peak_index[1]),
        correlation.shape[1] - np.abs(xx - peak_index[1]),
    )
    exclusion = (dy * dy + dx * dx) <= 25
    sidelobes = correlation[~exclusion]

    if sidelobes.size:
        mean = float(sidelobes.mean())
        std = float(sidelobes.std())
        psr = (peak - mean) / max(std, 1e-12)
        second = float(np.max(sidelobes))
    else:
        psr = 0.0
        second = 0.0

    peak_ratio = peak / max(abs(second), 1e-12)
    return {
        "shift_y": float(shift[0]),
        "shift_x": float(shift[1]),
        "peak": peak,
        "peak_ratio": float(peak_ratio),
        "psr": float(psr),
        "surface": correlation,
    }


def _centre_pad(
    image: np.ndarray,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    image = _rgb(image)
    h, w = image.shape[:2]
    canvas = np.zeros((side, side, 3), dtype=np.float64)
    mask = np.zeros((side, side), dtype=np.float64)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image
    mask[y0 : y0 + h, x0 : x0 + w] = 1.0
    return canvas, mask


def _resize_same_factor(
    image: np.ndarray,
    factor: float,
) -> np.ndarray:
    image = _rgb(image)
    if factor >= 0.999999:
        return image.copy()
    h, w = image.shape[:2]
    out_h = max(32, int(round(h * factor)))
    out_w = max(32, int(round(w * factor)))
    return dsp.resize(image, out_h, out_w, method="bilinear", antialias=True)


def prepare_pair(
    reference: np.ndarray,
    query: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Apply one common downscale factor, then centre-pad to a shared square canvas."""
    reference = _rgb(reference)
    images = [reference]
    if query is not None:
        query = _rgb(query)
        images.append(query)

    max_dimension = max(max(img.shape[:2]) for img in images)
    target_content = WORK_MAX_SIDE / 1.15
    factor = min(1.0, target_content / max_dimension)

    resized = [_resize_same_factor(img, factor) for img in images]
    max_after = max(max(img.shape[:2]) for img in resized)
    side = int(np.ceil(max_after * 1.15))
    side = max(64, min(WORK_MAX_SIDE, side))

    ref_canvas, ref_mask = _centre_pad(resized[0], side)
    if query is None:
        return ref_canvas, None, ref_mask, None

    query_canvas, query_mask = _centre_pad(resized[1], side)
    return ref_canvas, query_canvas, ref_mask, query_mask


def _normalised_correlation(
    reference: np.ndarray,
    query: np.ndarray,
    mask: np.ndarray,
) -> float:
    a = np.asarray(dsp.to_gray(reference), dtype=np.float64)[mask]
    b = np.asarray(dsp.to_gray(query), dtype=np.float64)[mask]
    if a.size < 16:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(a * b) / denominator)


def _masked_mse(
    reference: np.ndarray,
    query: np.ndarray,
    mask: np.ndarray,
) -> float:
    a = _rgb(reference)[mask]
    b = _rgb(query)[mask]
    if a.size == 0:
        return float("nan")
    return float(np.mean((a - b) ** 2))


def _display_correlation(surface: np.ndarray) -> np.ndarray:
    centred = np.fft.fftshift(np.asarray(surface, dtype=np.float64))
    positive = centred - float(np.percentile(centred, 1.0))
    positive = np.maximum(positive, 0.0)
    positive = np.log1p(positive * 200.0)
    return _normalise_display(positive, gamma=0.55)


def _checkerboard(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid: np.ndarray,
    block: int = 32,
) -> np.ndarray:
    reference = _rgb(reference)
    aligned = _rgb(aligned)
    yy, xx = np.indices(reference.shape[:2])
    use_query = ((yy // block + xx // block) % 2 == 1) & valid
    out = reference.copy()
    out[use_query] = aligned[use_query]
    return out


def _overlay(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    reference = _rgb(reference)
    aligned = _rgb(aligned)
    out = reference.copy()
    out[valid] = 0.5 * reference[valid] + 0.5 * aligned[valid]
    return np.clip(out, 0.0, 1.0)


def _difference(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    diff = np.abs(_rgb(reference) - _rgb(aligned))
    diff[~valid] = 0.0
    if np.any(valid):
        peak = float(np.percentile(diff[valid], 99.0))
        if peak > 1e-12:
            diff = np.clip(diff / peak, 0.0, 1.0)
    return diff


def _transform_translation_from_alignment_shift(
    angle_deg: float,
    scale: float,
    align_shift_x: float,
    align_shift_y: float,
) -> tuple[float, float]:
    """Recover the query's original translation from the post-R/S alignment shift."""
    theta = np.deg2rad(angle_deg)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    ax, ay = float(align_shift_x), float(align_shift_y)
    tx = -scale * (cosine * ax - sine * ay)
    ty = -scale * (sine * ax + cosine * ay)
    return float(tx), float(ty)


def register_similarity(
    reference: np.ndarray,
    query: np.ndarray,
    reference_mask: np.ndarray | None = None,
    query_mask: np.ndarray | None = None,
    use_hann: bool = True,
    subpixel: bool = True,
) -> dict:
    """Estimate query = scale*R(reference)+translation and align query to reference."""
    reference = _rgb(reference)
    query = _rgb(query)
    if reference.shape != query.shape:
        raise ValueError("Prepared reference and query canvases must have identical shapes")

    h, w = reference.shape[:2]
    if reference_mask is None:
        reference_mask = np.ones((h, w), dtype=np.float64)
    if query_mask is None:
        query_mask = np.ones((h, w), dtype=np.float64)

    ref_mag = _fft_magnitude(reference, use_hann)
    query_mag = _fft_magnitude(query, use_hann)
    ref_lp, log_radii = log_polar_transform(ref_mag)
    query_lp, _ = log_polar_transform(query_mag)

    rs_phase = phase_correlation(ref_lp, query_lp, subpixel=subpixel)
    angle_shift = rs_phase["shift_y"]
    radial_shift = rs_phase["shift_x"]

    angle = -angle_shift * 360.0 / ref_lp.shape[0]
    angle = ((angle + 180.0) % 360.0) - 180.0
    dlog = float(log_radii[1] - log_radii[0])
    scale = float(np.exp(radial_shift * dlog))
    # Radial phase correlation is cyclic, so unrelated images can produce a
    # wrapped shift corresponding to an absurd scale. Spectral Match is a
    # similarity-registration tool, not an unlimited zoom estimator.
    scale = float(np.clip(scale, 0.35, 3.0))

    # The real-image Fourier magnitude is centro-symmetric, so theta and
    # theta+180 degrees are indistinguishable at the first stage. Resolve that
    # ambiguity by trying both candidates and keeping the spatial registration
    # with the stronger normalized correlation (PSR breaks close ties).
    candidate_angles = [
        angle,
        ((angle + 180.0 + 180.0) % 360.0) - 180.0,
    ]
    candidates = []

    ref_gray = np.asarray(dsp.to_gray(reference), dtype=np.float64)
    if use_hann:
        ref_for_translation = ref_gray * _hann2(ref_gray.shape)
    else:
        ref_for_translation = ref_gray

    for candidate_angle in candidate_angles:
        corrected = warp_similarity(
            query,
            angle_deg=-candidate_angle,
            scale=1.0 / scale,
        )
        corrected_mask = warp_similarity(
            query_mask,
            angle_deg=-candidate_angle,
            scale=1.0 / scale,
        )
        corrected_gray = np.asarray(dsp.to_gray(corrected), dtype=np.float64)
        if use_hann:
            corrected_for_translation = corrected_gray * _hann2(corrected_gray.shape)
        else:
            corrected_for_translation = corrected_gray

        translation_phase = phase_correlation(
            ref_for_translation,
            corrected_for_translation,
            subpixel=subpixel,
        )
        aligned = warp_similarity(
            corrected,
            tx=translation_phase["shift_x"],
            ty=translation_phase["shift_y"],
        )
        aligned_mask = warp_similarity(
            corrected_mask,
            tx=translation_phase["shift_x"],
            ty=translation_phase["shift_y"],
        )
        valid = (reference_mask > 0.5) & (aligned_mask > 0.5)
        ncc = _normalised_correlation(reference, aligned, valid)

        candidates.append(
            {
                "angle": float(candidate_angle),
                "corrected": corrected,
                "translation_phase": translation_phase,
                "aligned": aligned,
                "aligned_mask": aligned_mask,
                "valid": valid,
                "ncc": ncc,
            }
        )

    best = max(candidates, key=lambda item: (item["ncc"], item["translation_phase"]["psr"]))
    translation_phase = best["translation_phase"]
    tx, ty = _transform_translation_from_alignment_shift(
        best["angle"],
        scale,
        translation_phase["shift_x"],
        translation_phase["shift_y"],
    )
    valid = best["valid"]
    mse = _masked_mse(reference, best["aligned"], valid)
    overlap = float(np.mean(valid))

    return {
        "angle": best["angle"],
        "scale": scale,
        "tx": tx,
        "ty": ty,
        "align_shift_x": translation_phase["shift_x"],
        "align_shift_y": translation_phase["shift_y"],
        "aligned": best["aligned"],
        "aligned_mask": best["aligned_mask"],
        "valid": valid,
        "ref_magnitude": ref_mag,
        "query_magnitude": query_mag,
        "ref_log_polar": ref_lp,
        "query_log_polar": query_lp,
        "rs_phase": rs_phase,
        "translation_phase": translation_phase,
        "ncc": best["ncc"],
        "mse": mse,
        "overlap": overlap,
    }


@register(
    "spectral_match",
    "Spectral Match",
    "Recover rotation, scale and translation with Fourier-Mellin registration and phase correlation.",
)
def op_spectral_match(image: np.ndarray, params: dict) -> OpResult:
    mode = _as_choice(params, "input_mode", {"controlled", "real"}, "controlled")
    use_hann = _as_bool(params, "use_hann", True)
    subpixel = _as_bool(params, "subpixel", True)

    truth_angle = _as_float(params, "rotation", 32.0, -170.0, 170.0)
    truth_scale = _as_float(params, "scale", 0.78, 0.5, 1.7)
    shift_x_fraction = _as_float(params, "shift_x_fraction", 0.10, -0.30, 0.30)
    shift_y_fraction = _as_float(params, "shift_y_fraction", -0.07, -0.30, 0.30)

    started = time.perf_counter()

    if mode == "real":
        query_source = params.get("_query_image")
        if query_source is None:
            raise ValueError("Upload a query image for two-image matching")
        reference, query, reference_mask, query_mask = prepare_pair(image, query_source)
        truth = None
    else:
        reference, _, reference_mask, _ = prepare_pair(image)
        side = reference.shape[0]
        truth_tx = shift_x_fraction * side
        truth_ty = shift_y_fraction * side
        query = warp_similarity(
            reference,
            angle_deg=truth_angle,
            scale=truth_scale,
            tx=truth_tx,
            ty=truth_ty,
        )
        query_mask = warp_similarity(
            reference_mask,
            angle_deg=truth_angle,
            scale=truth_scale,
            tx=truth_tx,
            ty=truth_ty,
        )
        truth = {
            "angle": truth_angle,
            "scale": truth_scale,
            "tx": truth_tx,
            "ty": truth_ty,
        }

    registration = register_similarity(
        reference,
        query,
        reference_mask=reference_mask,
        query_mask=query_mask,
        use_hann=use_hann,
        subpixel=subpixel,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    valid = registration["valid"]
    aligned = np.clip(registration["aligned"], 0.0, 1.0)
    overlay = _overlay(reference, aligned, valid)
    checker = _checkerboard(reference, aligned, valid)
    difference = _difference(reference, aligned, valid)

    ref_fft = _rgb(_normalise_display(registration["ref_magnitude"]))
    query_fft = _rgb(_normalise_display(registration["query_magnitude"]))
    ref_lp = _rgb(_normalise_display(registration["ref_log_polar"]))
    query_lp = _rgb(_normalise_display(registration["query_log_polar"]))
    rs_surface = _rgb(_display_correlation(registration["rs_phase"]["surface"]))
    t_surface = _rgb(_display_correlation(registration["translation_phase"]["surface"]))

    result = OpResult()
    result.panels = [
        Panel("reference", "Reference", reference, "Target coordinate system"),
        Panel(
            "query",
            "Query",
            query,
            "Controlled transform" if mode == "controlled" else "Second uploaded image",
        ),
        Panel(
            "reference_fft",
            "Reference FFT magnitude",
            ref_fft,
            "Centred log magnitude after optional Hann windowing",
        ),
        Panel(
            "query_fft",
            "Query FFT magnitude",
            query_fft,
            "Translation changes Fourier phase, so magnitude isolates rotation/scale",
        ),
        Panel(
            "reference_log_polar",
            "Reference log-polar",
            ref_lp,
            "Rotation becomes angular shift; scale becomes log-radius shift",
        ),
        Panel(
            "query_log_polar",
            "Query log-polar",
            query_lp,
            "Fourier magnitude resampled on (theta, log r)",
        ),
        Panel(
            "rotation_scale_correlation",
            "Rotation/scale phase correlation",
            rs_surface,
            "Correlation impulse in log-polar coordinates",
        ),
        Panel(
            "translation_correlation",
            "Translation phase correlation",
            t_surface,
            "Second correlation after undoing estimated rotation and scale",
        ),
        Panel(
            "aligned",
            "Registered query",
            aligned,
            "Query transformed into the reference coordinate system",
            download_image=aligned,
        ),
        Panel(
            "overlay",
            "Registered overlay",
            overlay,
            "50/50 reference-query blend inside their valid overlap",
        ),
        Panel(
            "checkerboard",
            "Checkerboard comparison",
            checker,
            "Alternating reference and aligned-query blocks expose local misregistration",
        ),
        Panel(
            "difference",
            "Registration difference",
            difference,
            "|reference - aligned query| within the valid overlap, auto-scaled",
        ),
    ]

    rs = registration["rs_phase"]
    tp = registration["translation_phase"]
    result.metrics = [
        Metric("Estimated rotation", f"{registration['angle']:+.3f}&deg;"),
        Metric("Estimated scale", f"{registration['scale']:.5f}&times;"),
        Metric("Estimated translation X", f"{registration['tx']:+.2f} px"),
        Metric("Estimated translation Y", f"{registration['ty']:+.2f} px"),
        Metric(
            "Rotation/scale PSR",
            f"{rs['psr']:.2f}",
            "Peak-to-sidelobe ratio of the log-polar phase-correlation impulse.",
        ),
        Metric("Rotation/scale peak ratio", f"{rs['peak_ratio']:.2f}&times;"),
        Metric(
            "Translation PSR",
            f"{tp['psr']:.2f}",
            "Peak-to-sidelobe ratio after rotation/scale correction.",
        ),
        Metric("Translation peak ratio", f"{tp['peak_ratio']:.2f}&times;"),
        Metric(
            "Aligned correlation",
            f"{registration['ncc']:.4f}",
            "Normalized correlation over pixels valid in both images; 1 is identical up to brightness offset/gain.",
        ),
        Metric("Overlap", f"{registration['overlap'] * 100:.1f}%"),
        Metric("Overlap MSE", _fmt(registration["mse"], 6)),
        Metric("Compute time", f"{elapsed_ms:.1f} ms"),
    ]

    if truth is not None:
        angle_error = abs(
            ((registration["angle"] - truth["angle"] + 180.0) % 360.0) - 180.0
        )
        scale_error = abs(registration["scale"] - truth["scale"])
        translation_error = float(
            np.hypot(
                registration["tx"] - truth["tx"],
                registration["ty"] - truth["ty"],
            )
        )
        result.metrics[4:4] = [
            Metric("True rotation", f"{truth['angle']:+.3f}&deg;"),
            Metric("Rotation error", f"{angle_error:.3f}&deg;"),
            Metric("True scale", f"{truth['scale']:.5f}&times;"),
            Metric("Scale error", f"{scale_error:.5f}"),
            Metric(
                "True translation",
                f"({truth['tx']:+.2f}, {truth['ty']:+.2f}) px",
            ),
            Metric("Translation error", f"{translation_error:.3f} px"),
        ]
        result.notes.append(
            "Controlled mode creates the query with a known similarity transform, then estimates "
            "that transform from the pixels alone. The true values are used only after registration "
            "to report errors; they are not passed into the estimator."
        )
    else:
        result.notes.append(
            "Two-image mode uses the second upload as the query. It assumes the two images show "
            "substantially the same planar content and differ mainly by uniform scale, in-plane "
            "rotation and translation. Strong perspective or different viewpoints violate that model."
        )

    result.notes.append(
        "Fourier magnitude removes translation because a spatial shift changes only Fourier phase. "
        "Log-polar sampling then converts rotation and uniform scale into translations, allowing the "
        "first phase-correlation stage to estimate both simultaneously."
    )
    result.notes.append(
        "Real-image Fourier magnitude has a 180-degree ambiguity. Spectral Match explicitly tests "
        "both candidate orientations and keeps the one whose spatially registered result has the "
        "stronger normalized correlation."
    )
    if use_hann:
        result.notes.append(
            "A 2-D Hann window was applied before the FFT to reduce artificial frequency energy "
            "from the rectangular image boundary."
        )
    if subpixel:
        result.notes.append(
            "Subpixel refinement fits a parabola through each phase-correlation peak and its two "
            "neighbors, producing fractional-pixel translation and finer rotation/scale estimates."
        )
    return result
