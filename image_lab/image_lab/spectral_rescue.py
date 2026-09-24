"""Periodic-obstruction suppression in the 2-D Fourier domain.

Spectral Rescue is intentionally a structured interference experiment, not a
claim that a single image can reveal arbitrary objects hidden behind opaque
foregrounds. Repetitive screens, scan lines, regular meshes and sinusoidal
banding create concentrated Fourier peaks. We detect those peaks, optionally
combine them with points selected by the user, build smooth symmetric notch
filters and reconstruct the image with an inverse FFT.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from . import dsp_utils as dsp
from .operations import (
    Metric,
    OpResult,
    Panel,
    _as_bool,
    _as_choice,
    _as_float,
    _as_int,
    _fmt,
    register,
)


SPECTRAL_PATTERNS = {
    "horizontal": "Horizontal bands",
    "vertical": "Vertical bands",
    "diagonal": "Diagonal bands",
    "grid": "Two-direction grid",
    "mesh": "Multi-frequency mesh",
}


def _grayscale(image: np.ndarray) -> np.ndarray:
    return np.asarray(
        dsp.to_gray(np.asarray(image, dtype=np.float64)), dtype=np.float64
    )


def _shifted_fft(plane: np.ndarray, remove_mean: bool = False) -> np.ndarray:
    work = np.asarray(plane, dtype=np.float64)
    if remove_mean:
        work = work - float(work.mean())
    return np.fft.fftshift(np.fft.fft2(work))


def _display_spectrum(shifted: np.ndarray) -> np.ndarray:
    """Convert a complex centred spectrum into a robust [0,1] display image."""
    log_mag = np.log1p(np.abs(shifted))
    if log_mag.size == 0:
        return np.zeros_like(log_mag, dtype=np.float64)
    lo = float(np.percentile(log_mag, 2.0))
    hi = float(np.percentile(log_mag, 99.7))
    if hi <= lo + 1e-12:
        return np.zeros_like(log_mag, dtype=np.float64)
    shown = np.clip((log_mag - lo) / (hi - lo), 0.0, 1.0)
    return np.sqrt(shown)


def _box_smooth(plane: np.ndarray, radius: int) -> np.ndarray:
    """Fast reflected local mean used as the spectral background estimate."""
    radius = max(1, int(radius))
    padded = np.pad(
        plane, ((radius, radius), (radius, radius)), mode="reflect"
    )
    integral = (
        np.pad(padded, ((1, 0), (1, 0)), mode="constant")
        .cumsum(0)
        .cumsum(1)
    )
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def add_periodic_interference(
    image: np.ndarray,
    pattern: str = "grid",
    frequency: float = 24.0,
    amplitude: float = 0.16,
    angle: float = 25.0,
) -> np.ndarray:
    """Add deterministic, DFT-friendly periodic interference to an image.

    Frequency is expressed as cycles across the image. Components are snapped
    to integer DFT bins so the controlled demonstration produces clear peaks
    rather than hiding the lesson behind spectral leakage.
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    xn = xx / max(w, 1)
    yn = yy / max(h, 1)
    theta = np.deg2rad(float(angle))
    frequency = max(1.0, float(frequency))

    def wave(direction: float, freq: float = frequency, phase: float = 0.0):
        fx = int(round(freq * np.cos(direction)))
        fy = int(round(freq * np.sin(direction)))
        if fx == 0 and fy == 0:
            fx = 1
        return np.cos(
            2.0 * np.pi * (fx * xn + fy * yn) + phase
        )

    if pattern == "horizontal":
        interference = wave(np.pi / 2.0)
    elif pattern == "vertical":
        interference = wave(0.0)
    elif pattern == "diagonal":
        interference = wave(theta)
    elif pattern == "mesh":
        interference = (
            0.45 * wave(theta)
            + 0.35 * wave(theta + np.pi / 2.0)
            + 0.20
            * wave(
                theta + np.deg2rad(37.0),
                frequency * 1.65,
                0.7,
            )
        )
    else:
        interference = 0.5 * (
            wave(theta) + wave(theta + np.pi / 2.0)
        )

    peak = float(np.max(np.abs(interference)))
    if peak > 1e-12:
        interference = interference / peak
    if image.ndim == 3:
        interference = interference[..., None]
    return np.clip(
        image + float(amplitude) * interference, 0.0, 1.0
    )


def detect_spectral_peak_pairs(
    image: np.ndarray,
    sensitivity: float = 6.0,
    ignore_center_fraction: float = 0.05,
    max_pairs: int = 5,
    require_symmetric: bool = True,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """Find narrow, locally exceptional peaks outside the low-frequency centre."""
    gray = _grayscale(image)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    shifted = _shifted_fft(gray, remove_mean=True)
    log_mag = np.log1p(np.abs(shifted))

    background_radius = min(
        25, max(4, int(round(min(h, w) * 0.02)))
    )
    background = _box_smooth(log_mag, background_radius)
    anomaly = log_mag - background

    yy, xx = np.ogrid[:h, :w]
    radius = float(ignore_center_fraction) * min(h, w)
    valid = (
        (yy - cy) ** 2 + (xx - cx) ** 2 >= radius ** 2
    )
    if h > 4 and w > 4:
        valid[:2, :] = False
        valid[-2:, :] = False
        valid[:, :2] = False
        valid[:, -2:] = False

    samples = anomaly[valid]
    if samples.size == 0:
        return [], anomaly
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    robust_sigma = max(1e-9, 1.4826 * mad)

    # Larger sensitivity means a lower anomaly threshold.
    z_threshold = 7.5 - 0.55 * np.clip(
        float(sensitivity), 1.0, 10.0
    )
    threshold = median + z_threshold * robust_sigma
    candidates = valid & (anomaly > threshold)

    # 8-neighbour local maxima without a large window allocation.
    maxima = candidates.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            maxima &= anomaly >= np.roll(
                anomaly, (dy, dx), axis=(0, 1)
            )

    ys, xs = np.where(maxima)
    if ys.size == 0:
        return [], anomaly
    scores = anomaly[ys, xs]
    order = np.argsort(scores)[::-1]

    chosen: list[tuple[int, int, float]] = []
    occupied: list[tuple[int, int]] = []
    min_separation = max(4.0, min(h, w) * 0.015)
    min_sep2 = min_separation ** 2

    for index in order:
        y, x = int(ys[index]), int(xs[index])
        my = (2 * cy - y) % h
        mx = (2 * cx - x) % w

        # Keep only one side of each conjugate pair.
        if (y, x) > (my, mx):
            continue
        if any(
            (y - py) ** 2 + (x - px) ** 2 < min_sep2
            for py, px in occupied
        ):
            continue

        if require_symmetric:
            mirror_score = float(anomaly[my, mx])
            if mirror_score < (
                median + 0.6 * (threshold - median)
            ):
                continue

        score = float(scores[index])
        chosen.append((y, x, score))
        occupied.extend([(y, x), (my, mx)])
        if len(chosen) >= int(max_pairs):
            break

    return chosen, anomaly


def _manual_pairs(
    raw_points: object,
    shape: tuple[int, int],
    limit: int = 16,
) -> list[tuple[int, int, float]]:
    if not isinstance(raw_points, list):
        return []
    h, w = shape
    points: list[tuple[int, int, float]] = []
    for item in raw_points[:limit]:
        if isinstance(item, dict):
            xv, yv = item.get("x"), item.get("y")
        elif (
            isinstance(item, (list, tuple))
            and len(item) >= 2
        ):
            xv, yv = item[0], item[1]
        else:
            continue
        try:
            x = float(xv)
            y = float(yv)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        px = int(round(x * max(w - 1, 0)))
        py = int(round(y * max(h - 1, 0)))
        points.append((py, px, float("inf")))
    return points


def _merge_pairs(
    automatic: Iterable[tuple[int, int, float]],
    manual: Iterable[tuple[int, int, float]],
    shape: tuple[int, int],
) -> list[tuple[int, int, float, str]]:
    h, w = shape
    cy, cx = h // 2, w // 2
    merged: list[tuple[int, int, float, str]] = []

    def add(
        y: int,
        x: int,
        score: float,
        source: str,
    ):
        my = (2 * cy - y) % h
        mx = (2 * cx - x) % w
        if (y, x) > (my, mx):
            y, x = my, mx
        for ey, ex, _, _ in merged:
            if (y - ey) ** 2 + (x - ex) ** 2 <= 9:
                return
        merged.append((y, x, score, source))

    for y, x, score in automatic:
        add(int(y), int(x), float(score), "auto")
    for y, x, score in manual:
        add(int(y), int(x), float(score), "manual")
    return merged


def build_notch_mask(
    shape: tuple[int, int],
    pairs: Iterable[
        tuple[int, int]
        | tuple[int, int, float]
        | tuple[int, int, float, str]
    ],
    radius: float = 4.0,
    softness: float = 1.0,
) -> np.ndarray:
    """Construct a smooth symmetric Gaussian notch-reject mask."""
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask = np.ones((h, w), dtype=np.float64)
    sigma = max(0.5, float(radius) * float(softness))

    for pair in pairs:
        y, x = int(pair[0]), int(pair[1])
        my = (2 * cy - y) % h
        mx = (2 * cx - x) % w
        for py, px in ((y, x), (my, mx)):
            d2 = (yy - py) ** 2 + (xx - px) ** 2
            mask *= 1.0 - np.exp(
                -d2 / (2.0 * sigma ** 2)
            )
    return np.clip(mask, 0.0, 1.0)


def apply_frequency_mask(
    image: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)

    def filter_plane(plane: np.ndarray):
        shifted = _shifted_fft(plane)
        restored = np.fft.ifft2(
            np.fft.ifftshift(shifted * mask)
        )
        return np.real(restored)

    if image.ndim == 2:
        return np.clip(filter_plane(image), 0.0, 1.0)
    channels = [
        filter_plane(image[..., c])
        for c in range(image.shape[-1])
    ]
    return np.clip(
        np.stack(channels, axis=-1), 0.0, 1.0
    )


def _draw_ring(
    rgb: np.ndarray,
    y: int,
    x: int,
    colour: np.ndarray,
    radius: int = 7,
):
    h, w = rgb.shape[:2]
    y0 = max(0, y - radius - 2)
    y1 = min(h, y + radius + 3)
    x0 = max(0, x - radius - 2)
    x1 = min(w, x + radius + 3)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt(
        (yy - y) ** 2 + (xx - x) ** 2
    )
    ring = np.abs(distance - radius) <= 1.25
    tile = rgb[y0:y1, x0:x1]
    tile[ring] = colour


def annotated_spectrum(
    shifted: np.ndarray,
    pairs: Iterable[tuple[int, int, float, str]],
    show_markers: bool = True,
) -> np.ndarray:
    shown = _display_spectrum(shifted)
    rgb = np.repeat(shown[..., None], 3, axis=2)
    if not show_markers:
        return rgb

    h, w = shown.shape
    cy, cx = h // 2, w // 2
    auto_colour = np.array([0.83, 0.96, 0.50])
    manual_colour = np.array([0.25, 0.90, 0.95])
    for y, x, _, source in pairs:
        colour = (
            manual_colour
            if source == "manual"
            else auto_colour
        )
        my = (2 * cy - y) % h
        mx = (2 * cx - x) % w
        _draw_ring(rgb, y, x, colour)
        _draw_ring(rgb, my, mx, colour)
    return rgb


def _frequency_summary(
    pair: tuple[int, int, float, str],
    shape: tuple[int, int],
) -> str:
    y, x, _, source = pair
    h, w = shape
    cy, cx = h // 2, w // 2
    fy = (y - cy) / float(h)
    fx = (x - cx) / float(w)
    radial = float(np.hypot(fx, fy))
    period = (
        float("inf")
        if radial <= 1e-12
        else 1.0 / radial
    )
    angle = float(np.degrees(np.arctan2(fy, fx)))
    return (
        f"{source}: fx={fx:+.4f}, fy={fy:+.4f} cycles/pixel, "
        f"period≈{period:.1f}px, angle={angle:+.1f}&deg;"
    )


@register(
    "spectral_rescue",
    "Spectral Rescue",
    "Detect repetitive visual interference in the 2-D Fourier spectrum and suppress it with smooth notch filters.",
)
def op_spectral_rescue(
    image: np.ndarray, params: dict
) -> OpResult:
    input_mode = _as_choice(
        params,
        "input_mode",
        {"uploaded", "simulate"},
        "simulate",
    )
    auto_detect = _as_bool(
        params, "auto_detect", True
    )
    sensitivity = _as_float(
        params, "sensitivity", 6.0, 1.0, 10.0
    )
    ignore_center = _as_float(
        params, "ignore_center", 0.05, 0.01, 0.30
    )
    max_pairs = _as_int(
        params, "max_pairs", 5, 1, 12
    )
    require_symmetric = _as_bool(
        params, "require_symmetric", True
    )
    notch_radius = _as_float(
        params, "notch_radius", 4.0, 0.75, 24.0
    )
    notch_softness = _as_float(
        params, "notch_softness", 1.0, 0.25, 3.0
    )
    show_markers = _as_bool(
        params, "show_markers", True
    )

    pattern = _as_choice(
        params,
        "pattern",
        set(SPECTRAL_PATTERNS),
        "grid",
    )
    pattern_frequency = _as_float(
        params, "pattern_frequency", 24.0, 2.0, 96.0
    )
    pattern_amplitude = _as_float(
        params, "pattern_amplitude", 0.16, 0.0, 0.45
    )
    pattern_angle = _as_float(
        params, "pattern_angle", 25.0, -90.0, 90.0
    )

    started = time.perf_counter()
    if input_mode == "simulate":
        observed = add_periodic_interference(
            image,
            pattern=pattern,
            frequency=pattern_frequency,
            amplitude=pattern_amplitude,
            angle=pattern_angle,
        )
    else:
        observed = np.asarray(
            image, dtype=np.float64
        ).copy()

    gray = _grayscale(observed)
    shifted_gray = _shifted_fft(gray)

    automatic: list[tuple[int, int, float]] = []
    if auto_detect:
        automatic, _ = detect_spectral_peak_pairs(
            observed,
            sensitivity=sensitivity,
            ignore_center_fraction=ignore_center,
            max_pairs=max_pairs,
            require_symmetric=require_symmetric,
        )
    manual = _manual_pairs(
        params.get("manual_peaks", []), gray.shape
    )
    pairs = _merge_pairs(
        automatic, manual, gray.shape
    )

    mask = build_notch_mask(
        gray.shape,
        pairs,
        radius=notch_radius,
        softness=notch_softness,
    )
    restored = apply_frequency_mask(observed, mask)
    filtered_shifted = shifted_gray * mask

    energy_before = float(
        np.sum(np.abs(shifted_gray) ** 2)
    )
    energy_after = float(
        np.sum(np.abs(filtered_shifted) ** 2)
    )
    removed_energy = (
        0.0
        if energy_before <= 1e-20
        else max(
            0.0,
            1.0 - energy_after / energy_before,
        )
    )
    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000.0

    spectrum_rgb = annotated_spectrum(
        shifted_gray,
        pairs,
        show_markers=show_markers,
    )
    filtered_rgb = np.repeat(
        _display_spectrum(filtered_shifted)[..., None],
        3,
        axis=2,
    )
    mask_rgb = np.repeat(
        mask[..., None], 3, axis=2
    )
    suppressed = dsp.difference_map(
        observed, restored, gain="auto"
    )

    result = OpResult()
    if input_mode == "simulate":
        result.panels.extend(
            [
                Panel(
                    "original",
                    "Original",
                    image,
                    "Clean reference",
                ),
                Panel(
                    "observed",
                    "Periodic interference",
                    observed,
                    f"{SPECTRAL_PATTERNS[pattern]} · {pattern_frequency:.1f} cycles/image",
                ),
            ]
        )
    else:
        result.panels.append(
            Panel(
                "observed",
                "Uploaded observation",
                observed,
                "Image containing suspected repetitive interference",
            )
        )

    result.panels.extend(
        [
            Panel(
                "spectrum",
                "Fourier spectrum",
                spectrum_rgb,
                "Centred log magnitude. Green rings = automatic peaks; cyan rings = manual selections.",
            ),
            Panel(
                "mask",
                "Notch-reject mask",
                mask_rgb,
                "White is preserved; dark holes suppress selected frequency pairs.",
            ),
            Panel(
                "restored",
                "Spectral rescue",
                restored,
                "IFFT reconstruction after symmetric notch filtering.",
            ),
            Panel(
                "filtered_spectrum",
                "Filtered spectrum",
                filtered_rgb,
                "Selected concentrated peaks are attenuated while the rest of the spectrum is retained.",
            ),
            Panel(
                "suppressed",
                "Suppressed component",
                suppressed,
                "|observed - restored|, auto-scaled to show what the notches removed.",
            ),
        ]
    )

    auto_count = sum(
        1 for p in pairs if p[3] == "auto"
    )
    manual_count = sum(
        1 for p in pairs if p[3] == "manual"
    )
    result.metrics = [
        Metric("Automatic peak pairs", str(auto_count)),
        Metric("Manual peak pairs", str(manual_count)),
        Metric("Total notch pairs", str(len(pairs))),
        Metric(
            "Sensitivity",
            f"{sensitivity:.1f} / 10",
        ),
        Metric(
            "Ignored centre radius",
            f"{ignore_center * 100:.1f}% of min dimension",
        ),
        Metric(
            "Notch radius",
            f"{notch_radius:.1f} px",
        ),
        Metric(
            "Notch softness",
            f"{notch_softness:.2f}&times;",
        ),
        Metric(
            "Spectral energy removed",
            f"{removed_energy * 100:.3f}%",
            "Energy removed from the observed luminance spectrum by the notch mask.",
        ),
        Metric(
            "FFT analysis time",
            f"{elapsed_ms:.1f} ms",
        ),
    ]

    if input_mode == "simulate":
        psnr_bad = dsp.psnr(image, observed)
        psnr_restored = dsp.psnr(image, restored)
        ssim_bad = dsp.ssim(image, observed)
        ssim_restored = dsp.ssim(image, restored)
        result.metrics.extend(
            [
                Metric(
                    "PSNR corrupted",
                    f"{_fmt(psnr_bad, 2)} dB",
                ),
                Metric(
                    "PSNR restored",
                    f"{_fmt(psnr_restored, 2)} dB",
                ),
                Metric(
                    "PSNR gain",
                    f"{psnr_restored - psnr_bad:+.2f} dB",
                ),
                Metric(
                    "SSIM corrupted",
                    _fmt(ssim_bad, 4),
                ),
                Metric(
                    "SSIM restored",
                    _fmt(ssim_restored, 4),
                ),
            ]
        )
        result.notes.append(
            "Controlled mode adds known periodic interference to the uploaded clean image. "
            "Because the clean reference is still available, PSNR and SSIM measure whether "
            "the frequency-domain reconstruction actually recovered information."
        )
    else:
        result.notes.append(
            "Uploaded mode assumes the image already contains periodic interference. "
            "There is no clean reference, so PSNR/SSIM would be fake and are intentionally omitted."
        )

    if not pairs:
        result.notes.append(
            "No notch pairs are active, so the reconstruction is effectively unchanged. "
            "Increase sensitivity or click a bright off-centre peak in the Fourier spectrum."
        )
    else:
        result.notes.append(
            "Click the Fourier-spectrum panel to toggle manual notch locations. "
            "Each click automatically includes its conjugate-symmetric partner so the inverse FFT remains real."
        )
        summaries = [
            _frequency_summary(pair, gray.shape)
            for pair in pairs[:8]
        ]
        result.notes.append(
            "Selected frequency pairs:<br>"
            + "<br>".join(summaries)
        )

    result.notes.append(
        "This method targets repetitive or near-periodic interference such as screens, grids, "
        "scan lines and regular banding. It cannot reconstruct scene content that an opaque, "
        "irregular foreground object has completely hidden."
    )
    return result
