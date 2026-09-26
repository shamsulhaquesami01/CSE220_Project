"""Frequency-domain motion deblurring experiment."""

from __future__ import annotations

import time

import numpy as np

from . import dsp_utils as dsp
from .operations import (
    Metric,
    OpResult,
    Panel,
    _as_choice,
    _as_float,
    _as_int,
    _fmt,
    register,
)


@register(
    "deblur",
    "FFT Deblur",
    "Deblur an uploaded image or simulate camera shake for a controlled comparison.",
)
def op_deblur(image: np.ndarray, params: dict) -> OpResult:
    input_mode = _as_choice(
        params, "input_mode", {"uploaded", "simulate"}, "simulate"
    )
    motion_length = _as_float(params, "motion_length", 17.0, 1.0, 61.0)
    motion_angle = _as_float(params, "motion_angle", 18.0, -90.0, 90.0)
    noise_sigma = _as_float(params, "noise_sigma", 0.012, 0.0, 0.20)
    wiener_log10 = _as_float(params, "wiener_log10", -3.0, -8.0, -0.5)
    inverse_floor_log10 = _as_float(
        params, "inverse_floor_log10", -3.0, -7.0, -1.0
    )
    seed = _as_int(params, "seed", 0, 0, 10_000_000)

    wiener_k = 10.0 ** wiener_log10
    inverse_floor = 10.0 ** inverse_floor_log10

    started = time.perf_counter()
    if input_mode == "uploaded":
        inverse, wiener, psf = dsp.motion_deblur_observation(
            image,
            length=motion_length,
            angle=motion_angle,
            wiener_k=wiener_k,
            inverse_floor=inverse_floor,
        )
    else:
        blurred, degraded, inverse, wiener, psf = dsp.motion_deblur_experiment(
            image,
            length=motion_length,
            angle=motion_angle,
            noise_sigma=noise_sigma,
            wiener_k=wiener_k,
            inverse_floor=inverse_floor,
            seed=seed,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Show the blur transfer function.
    spectrum = dsp.transfer_magnitude_image(psf, image.shape[:2])
    spectrum_rgb = np.repeat(spectrum[..., None], 3, axis=2)

    h = dsp.psf_to_otf(psf, image.shape[:2])
    abs_h = np.abs(h)
    near_zero_fraction = float(np.mean(abs_h < 0.02))

    result = OpResult()
    if input_mode == "uploaded":
        result.panels = [
            Panel("observed", "Uploaded blurred image", image, "Observed image g[m,n]"),
        ]
    else:
        result.panels = [
            Panel("original", "Original", image, "Reference f[m,n]"),
            Panel(
                "degraded",
                "Camera-shake model",
                degraded,
                f"Motion blur: {motion_length:.1f}px at {motion_angle:.1f}&deg; + Gaussian noise",
            ),
        ]
    result.panels.extend([
        Panel(
            "inverse",
            "Direct inverse",
            inverse,
            "F-hat = G / H; unstable near zeros of H",
        ),
        Panel(
            "wiener",
            "Wiener restoration",
            wiener,
            "F-hat = G H* / (|H|^2 + K)",
        ),
        Panel(
            "transfer",
            "Blur transfer |H(u,v)|",
            spectrum_rgb,
            "Centred log-magnitude of the motion-blur transfer function",
        ),
    ])

    result.metrics = [
        Metric("Motion length", f"{motion_length:.1f} px"),
        Metric("Motion angle", f"{motion_angle:.1f}&deg;"),
        Metric(
            "Wiener K",
            f"{wiener_k:.2e}",
            "Larger K suppresses unstable frequencies more strongly but also smooths detail.",
        ),
        Metric(
            "Inverse cutoff",
            f"{inverse_floor:.1e}",
            "Only true/near spectral zeros are suppressed; small nonzero H still amplifies noise.",
        ),
        Metric(
            "Near-zero H bins",
            f"{near_zero_fraction * 100:.2f}%",
            "Fraction of transfer-function bins with |H| < 0.02.",
        ),
        Metric("FFT restoration time", f"{elapsed_ms:.1f} ms"),
    ]

    if input_mode == "simulate":
        psnr_degraded = dsp.psnr(image, degraded)
        psnr_inverse = dsp.psnr(image, inverse)
        psnr_wiener = dsp.psnr(image, wiener)
        ssim_degraded = dsp.ssim(image, degraded)
        ssim_inverse = dsp.ssim(image, inverse)
        ssim_wiener = dsp.ssim(image, wiener)
        result.metrics[2:2] = [Metric("Noise &sigma;", _fmt(noise_sigma, 4))]
        result.metrics.extend([
            Metric("PSNR degraded", f"{_fmt(psnr_degraded, 2)} dB"),
            Metric("PSNR inverse", f"{_fmt(psnr_inverse, 2)} dB"),
            Metric("PSNR Wiener", f"{_fmt(psnr_wiener, 2)} dB"),
            Metric("SSIM degraded", _fmt(ssim_degraded, 4)),
            Metric("SSIM inverse", _fmt(ssim_inverse, 4)),
            Metric("SSIM Wiener", _fmt(ssim_wiener, 4)),
        ])

    if input_mode == "uploaded":
        result.notes.append(
            "The uploaded image is treated as the observed blurred signal G. No extra blur or "
            "noise is added. Because there is no clean reference image, PSNR and SSIM cannot "
            "be calculated for this mode."
        )
        result.notes.append(
            "This is non-blind deconvolution: motion length and angle describe the blur the "
            "camera applied. Adjust them until edges align and ringing is minimised."
        )
    else:
        result.notes.append(
            "The blur is not just a visual effect: a motion point-spread function h[m,n] is "
            "transformed to H(u,v), the image is transformed to F(u,v), and degradation is "
            "created by G = H F + N. The restoration therefore works on complex FFT bins."
        )
    result.notes.append(
        "Direct inverse filtering divides by H. Wherever the motion blur has almost erased a "
        "frequency (|H| is tiny), the same division magnifies noise enormously. The striped "
        "dark regions in the transfer-function panel show where that instability comes from."
    )
    result.notes.append(
        "The Wiener filter replaces raw division with H* / (|H|^2 + K). K is the "
        "regularisation/noise trade-off: too small behaves like the unstable inverse; too "
        "large is stable but over-smooths the reconstruction."
    )

    if input_mode == "simulate" and psnr_wiener > psnr_inverse:
        result.notes.append(
            f"For these settings Wiener gains {psnr_wiener - psnr_inverse:.2f} dB over "
            "the direct inverse, showing why regularisation matters when noise is present."
        )
    elif input_mode == "simulate" and noise_sigma == 0.0:
        result.notes.append(
            "With no added noise, direct inversion can be competitive because there is less "
            "noise for the near-zero frequencies to amplify. Increase noise to expose the "
            "ill-conditioned inverse problem."
        )

    return result
