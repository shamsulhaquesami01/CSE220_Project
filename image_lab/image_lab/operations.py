"""DSP operation registry and experiment handlers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import dsp_utils as dsp


# Result containers


@dataclass
class Panel:
    """One image tile in the results grid."""

    key: str
    label: str
    image: np.ndarray
    caption: str = ""
    download_image: np.ndarray | None = None


@dataclass
class Metric:
    """One row in the measurements table."""

    label: str
    value: str
    hint: str = ""


@dataclass
class OpResult:
    panels: list[Panel] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Operation:
    op_id: str
    label: str
    description: str
    handler: Callable[[np.ndarray, dict], OpResult]


OPERATIONS: dict[str, Operation] = {}


# Register an operation.
def register(op_id: str, label: str, description: str):
    def decorator(func):
        OPERATIONS[op_id] = Operation(op_id, label, description, func)
        return func

    return decorator


# Parse and clamp request parameters.


def _as_float(params, key, default, low=None, high=None):
    try:
        value = float(params.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if not np.isfinite(value):
        value = float(default)
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _as_int(params, key, default, low=None, high=None):
    return int(round(_as_float(params, key, default, low, high)))


def _as_bool(params, key, default=False):
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_choice(params, key, choices, default):
    value = str(params.get(key, default))
    return value if value in choices else default


# Parse the kernel grid.
def _as_kernel(params, key="kernel"):
    raw = params.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError("Kernel must be a non-empty list of rows")

    rows = []
    width = None
    for row in raw:
        if not isinstance(row, list) or not row:
            raise ValueError("Every kernel row must be a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("All kernel rows must have the same length")

        parsed = []
        for cell in row:
            try:
                number = float(cell)
            except (TypeError, ValueError):
                number = 0.0
            parsed.append(0.0 if not np.isfinite(number) else number)
        rows.append(parsed)

    kernel = np.array(rows, dtype=np.float64)
    if kernel.shape[0] > 15 or kernel.shape[1] > 15:
        raise ValueError("Kernel is limited to 15x15 to keep the preview responsive")
    return kernel


def _fmt(value, digits=4):
    if value == float("inf"):
        return "&infin;"
    return f"{value:.{digits}f}"


# 2D convolution


@register(
    "convolve",
    "Convolution",
    "Slide an editable kernel over the image and see the response change live.",
)
def op_convolve(image: np.ndarray, params: dict) -> OpResult:
    kernel = _as_kernel(params)
    pad_mode = _as_choice(params, "pad_mode", set(dsp.PAD_MODES), "reflect")
    normalize = _as_bool(params, "normalize", False)

    raw_sum = float(kernel.sum())
    if normalize:
        kernel = dsp.normalize_kernel(kernel)

    started = time.perf_counter()
    filtered_raw = dsp.convolve2d(image, kernel, pad_mode)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Keep raw values for range and clipping stats.
    out_min = float(filtered_raw.min())
    out_max = float(filtered_raw.max())
    filtered = np.clip(filtered_raw, 0.0, 1.0)

    clipped_fraction = float(np.mean((filtered_raw < 0.0) | (filtered_raw > 1.0)))

    result = OpResult()
    result.panels = [
        Panel("original", "Original", image, "Input f[m, n]"),
        Panel("filtered", "Filtered", filtered, "Output g = f * h"),
        Panel(
            "difference",
            "Difference",
            dsp.difference_map(image, filtered, gain="auto"),
            "|f - g|, auto-scaled",
        ),
    ]

    kernel_sum = float(kernel.sum())
    result.metrics = [
        Metric("Kernel size", f"{kernel.shape[0]} &times; {kernel.shape[1]}"),
        Metric(
            "Kernel sum (DC gain)",
            _fmt(kernel_sum),
            "1.0 preserves average brightness; 0.0 marks a pure high-pass / edge kernel.",
        ),
        Metric("MSE vs original", _fmt(dsp.mse(image, filtered), 6)),
        Metric("PSNR vs original", f"{_fmt(dsp.psnr(image, filtered), 2)} dB"),
        Metric("Raw output range", f"[{_fmt(out_min, 3)}, {_fmt(out_max, 3)}]"),
        Metric("Clipped pixels", f"{clipped_fraction * 100:.2f}%"),
        Metric("Compute time", f"{elapsed_ms:.1f} ms"),
    ]

    if normalize and not np.isclose(raw_sum, 0.0):
        result.notes.append(
            f"Kernel was normalised: coefficients divided by {raw_sum:.4f} so the sum is 1."
        )
    if np.isclose(kernel_sum, 0.0):
        result.notes.append(
            "This kernel sums to zero, so it removes the DC component. Flat regions map to "
            "0 and only edges survive; most of the response is negative and gets clipped to "
            "black for display."
        )
    elif kernel_sum > 1.5:
        result.notes.append(
            f"Kernel sum is {kernel_sum:.2f}, above 1. The image will brighten overall."
        )
    if clipped_fraction > 0.02:
        result.notes.append(
            f"{clipped_fraction * 100:.1f}% of samples fell outside [0, 1] and were clipped. "
            "Sharpening kernels overshoot at edges -- this is the Gibbs-like halo you can see."
        )
    return result


# Resampling with anti-aliasing


@register(
    "resample",
    "Resize & Anti-alias",
    "Downsample with and without a low-pass prefilter and compare the aliasing.",
)
def op_resample(image: np.ndarray, params: dict) -> OpResult:
    in_h, in_w = image.shape[:2]

    scale = _as_float(params, "scale", 0.25, 0.02, 4.0)
    method = _as_choice(params, "method", {"nearest", "bilinear"}, "bilinear")
    pad_mode = _as_choice(params, "pad_mode", set(dsp.PAD_MODES), "reflect")

    out_h = max(1, int(round(in_h * scale)))
    out_w = max(1, int(round(in_w * scale)))

    started = time.perf_counter()
    without_aa = dsp.resize(image, out_h, out_w, method, antialias=False, pad_mode=pad_mode)
    with_aa = dsp.resize(image, out_h, out_w, method, antialias=True, pad_mode=pad_mode)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    sigma_y = dsp.antialias_sigma(in_h, out_h)
    sigma_x = dsp.antialias_sigma(in_w, out_w)

    # Upscale previews with nearest-neighbor for a fair comparison.
    view_no_aa = dsp.resize_nearest(without_aa, in_h, in_w)
    view_aa = dsp.resize_nearest(with_aa, in_h, in_w)

    difference = dsp.difference_map(view_no_aa, view_aa, gain="auto")
    alias_energy = float(np.var(view_no_aa.astype(np.float64) - view_aa.astype(np.float64)))

    result = OpResult()
    result.panels = [
        Panel("original", "Original", image, f"{in_w} &times; {in_h}"),
        Panel(
            "no_aa",
            "Without anti-aliasing",
            view_no_aa,
            f"Resized to {out_w} &times; {out_h}; preview at source size",
            download_image=without_aa,
        ),
        Panel(
            "aa",
            "With anti-aliasing",
            view_aa,
            f"Resized to {out_w} &times; {out_h}; "
            + ("Gaussian prefilter applied" if scale < 1.0 else "No prefilter needed")
            + "; preview at source size",
            download_image=with_aa,
        ),
        Panel("difference", "Difference", difference, "Alias energy, auto-scaled"),
    ]

    decimation = 1.0 / scale if scale > 0 else float("inf")
    result.metrics = [
        Metric("Source size", f"{in_w} &times; {in_h}"),
        Metric("Target size", f"{out_w} &times; {out_h}"),
        Metric("Scale factor", f"{scale:.3f}&times;"),
        Metric(
            "Decimation factor D",
            f"{decimation:.3f}",
            "New Nyquist limit is &pi;/D on the original sample grid.",
        ),
        Metric(
            "Prefilter &sigma;",
            f"x: {sigma_x:.3f}, y: {sigma_y:.3f}",
            "&sigma; = (D - 1) / 2, and 0 when upsampling.",
        ),
        Metric("Interpolation", method),
        Metric("MSE (no-AA vs AA)", _fmt(dsp.mse(view_no_aa, view_aa), 6)),
        Metric("PSNR (no-AA vs AA)", f"{_fmt(dsp.psnr(view_no_aa, view_aa), 2)} dB"),
        Metric(
            "Alias energy",
            _fmt(alias_energy, 6),
            "Variance of the difference. Higher means more folded-back content.",
        ),
        Metric("Compute time", f"{elapsed_ms:.1f} ms"),
    ]

    if scale >= 1.0:
        result.notes.append(
            "You are upsampling, so there is no new Nyquist limit to violate and no "
            "prefilter is applied -- the two results are identical by construction. "
            "Drop the scale below 1.0 to see anti-aliasing do work."
        )
    else:
        result.notes.append(
            f"Keeping every {decimation:.2f}nd sample drops the Nyquist limit to "
            f"&pi;/{decimation:.2f}. Detail above that limit cannot be represented; without "
            "the prefilter it folds back into the baseband as moire instead of disappearing."
        )
        if method == "nearest":
            result.notes.append(
                "Nearest-neighbour sampling aliases hardest because it takes one raw sample "
                "per output pixel and averages nothing. Switch to bilinear to see the "
                "interpolator provide a little implicit low-pass on its own."
            )
    return result


# Noise and restoration

NOISE_MODELS = {
    "none": "No noise",
    "gaussian": "Additive white Gaussian",
    "salt_pepper": "Salt & pepper (impulse)",
}

CLEAN_FILTERS = {
    "none": "No filter",
    "mean": "Moving average",
    "median": "Median",
    "gaussian": "Gaussian low-pass",
}


@register(
    "noise",
    "Noise & Cleaning",
    "Corrupt the image with a known noise model, then compare restoration filters.",
)
def op_noise(image: np.ndarray, params: dict) -> OpResult:
    noise_model = _as_choice(params, "noise_model", set(NOISE_MODELS), "salt_pepper")
    clean_filter = _as_choice(params, "clean_filter", set(CLEAN_FILTERS), "median")
    pad_mode = _as_choice(params, "pad_mode", set(dsp.PAD_MODES), "reflect")

    noise_sigma = _as_float(params, "noise_sigma", 0.08, 0.0, 1.0)
    noise_amount = _as_float(params, "noise_amount", 0.06, 0.0, 1.0)
    filter_size = _as_int(params, "filter_size", 3, 1, 15)
    filter_sigma = _as_float(params, "filter_sigma", 1.0, 0.1, 10.0)
    # Keep the same noise while tuning filters.
    seed = _as_int(params, "seed", 0, 0, 10_000_000)

    # Use an odd filter window.
    if filter_size % 2 == 0:
        filter_size += 1

    if noise_model == "gaussian":
        noisy = dsp.add_gaussian_noise(image, noise_sigma, seed=seed)
    elif noise_model == "salt_pepper":
        noisy = dsp.add_salt_pepper(image, noise_amount, seed=seed)
    else:
        noisy = image.copy()

    started = time.perf_counter()
    if clean_filter == "mean":
        cleaned = dsp.mean_filter2d(noisy, filter_size, pad_mode)
    elif clean_filter == "median":
        cleaned = dsp.median_filter2d(noisy, filter_size, pad_mode)
    elif clean_filter == "gaussian":
        cleaned = dsp.gaussian_blur(noisy, filter_sigma, pad_mode)
    else:
        cleaned = noisy.copy()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    cleaned = np.clip(cleaned, 0.0, 1.0)

    psnr_noisy = dsp.psnr(image, noisy)
    psnr_clean = dsp.psnr(image, cleaned)
    gain_db = psnr_clean - psnr_noisy

    result = OpResult()
    result.panels = [
        Panel("original", "Original", image, "Clean reference"),
        Panel("noisy", "Noisy", noisy, NOISE_MODELS[noise_model]),
        Panel("cleaned", "Cleaned", cleaned, CLEAN_FILTERS[clean_filter]),
        Panel(
            "residual",
            "Residual",
            dsp.difference_map(image, cleaned, gain="auto"),
            "|original - cleaned|, auto-scaled",
        ),
    ]

    result.metrics = [
        Metric("Noise model", NOISE_MODELS[noise_model]),
        Metric("Cleaning filter", CLEAN_FILTERS[clean_filter]),
        Metric("MSE noisy", _fmt(dsp.mse(image, noisy), 6)),
        Metric("MSE cleaned", _fmt(dsp.mse(image, cleaned), 6)),
        Metric("PSNR noisy", f"{_fmt(psnr_noisy, 2)} dB"),
        Metric("PSNR cleaned", f"{_fmt(psnr_clean, 2)} dB"),
        Metric(
            "PSNR gain",
            f"{gain_db:+.2f} dB",
            "Positive means the filter recovered more than it destroyed.",
        ),
        Metric("SSIM noisy", _fmt(dsp.ssim(image, noisy), 4)),
        Metric("SSIM cleaned", _fmt(dsp.ssim(image, cleaned), 4)),
        Metric("Compute time", f"{elapsed_ms:.1f} ms"),
    ]

    if noise_model == "salt_pepper" and clean_filter == "mean":
        result.notes.append(
            "A moving average is the wrong tool for impulse noise: it cannot discard an "
            "outlier, it only spreads each spike over the whole window. Switch to the "
            "median to see an order-statistic filter reject the impulses outright."
        )
    if noise_model == "gaussian" and clean_filter == "median":
        result.notes.append(
            "The median does work on Gaussian noise, but a linear filter is a better match "
            "here: Gaussian noise is not made of outliers, so averaging is close to optimal."
        )
    if gain_db < 0:
        result.notes.append(
            "PSNR went down. The filter is blurring away more real signal than it is "
            "removing noise -- try a smaller window."
        )
    return result


# Look up an operation.
def get_operation(op_id: str) -> Operation:
    return OPERATIONS[op_id]
