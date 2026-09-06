"""Core NumPy DSP routines used by the Image Lab."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PAD_MODES = {
    "reflect": "reflect",
    "edge": "edge",
    "wrap": "wrap",
    "zero": "constant",
}


def to_float(image):
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.astype(np.float64) / 255.0
    return np.clip(array.astype(np.float64), 0.0, 1.0)


def to_uint8(image):
    return np.clip(np.asarray(image) * 255.0, 0.0, 255.0).round().astype(np.uint8)


def _per_channel(func, image, *args, **kwargs):
    image = np.asarray(image)
    if image.ndim == 2:
        return func(image, *args, **kwargs)
    return np.stack(
        [func(image[..., c], *args, **kwargs) for c in range(image.shape[-1])],
        axis=-1,
    )


def to_gray(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    return image[..., :3] @ np.array([0.299, 0.587, 0.114])


def _anchor_padding(kh, kw):
    top = kh // 2
    left = kw // 2
    return top, kh - 1 - top, left, kw - 1 - left


def _convolve2d_plane(plane, kernel, pad_mode="reflect"):
    kh, kw = kernel.shape
    top, bottom, left, right = _anchor_padding(kh, kw)
    flipped = kernel[::-1, ::-1]
    np_mode = PAD_MODES.get(pad_mode, "reflect")

    if np_mode == "constant":
        padded = np.pad(
            plane,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        padded = np.pad(plane, ((top, bottom), (left, right)), mode=np_mode)

    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("mnuv,uv->mn", windows, flipped, optimize=True)


def convolve2d(image, kernel, pad_mode="reflect"):
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array")
    if kernel.size == 0:
        raise ValueError("Kernel must not be empty")
    return _per_channel(
        _convolve2d_plane,
        np.asarray(image, dtype=np.float64),
        kernel,
        pad_mode,
    )


def _convolve_separable_plane(plane, kx, ky, pad_mode="reflect"):
    out = plane
    if kx.size > 1 or not np.isclose(kx[0], 1.0):
        out = _convolve2d_plane(out, kx.reshape(1, -1), pad_mode)
    if ky.size > 1 or not np.isclose(ky[0], 1.0):
        out = _convolve2d_plane(out, ky.reshape(-1, 1), pad_mode)
    return out


def convolve_separable(image, kx, ky, pad_mode="reflect"):
    kx = np.asarray(kx, dtype=np.float64).ravel()
    ky = np.asarray(ky, dtype=np.float64).ravel()
    return _per_channel(
        _convolve_separable_plane,
        np.asarray(image, dtype=np.float64),
        kx,
        ky,
        pad_mode,
    )


def gaussian_kernel1d(sigma, radius=None):
    if sigma <= 0:
        return np.array([1.0])
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_kernel2d(sigma, size=None):
    radius = None if size is None else max(1, int(size) // 2)
    k1d = gaussian_kernel1d(sigma, radius)
    return np.outer(k1d, k1d)


def box_kernel(n):
    n = max(1, int(n))
    return np.full((n, n), 1.0 / (n * n), dtype=np.float64)


def normalize_kernel(kernel):
    kernel = np.asarray(kernel, dtype=np.float64)
    total = kernel.sum()
    if np.isclose(total, 0.0):
        return kernel
    return kernel / total


KERNEL_PRESETS = {
    "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    "box_blur_3": [
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
    ],
    "gaussian_3": [
        [1 / 16, 2 / 16, 1 / 16],
        [2 / 16, 4 / 16, 2 / 16],
        [1 / 16, 2 / 16, 1 / 16],
    ],
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "sharpen_strong": [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
    "laplacian": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    "sobel_x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "sobel_y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
}


def unsharp_mask(image, sigma=1.0, amount=1.0, pad_mode="reflect"):
    k1d = gaussian_kernel1d(sigma)
    blurred = convolve_separable(image, k1d, k1d, pad_mode)
    detail = np.asarray(image, dtype=np.float64) - blurred
    return np.asarray(image, dtype=np.float64) + amount * detail


def _sample_coords(in_size, out_size):
    return (
        (np.arange(out_size, dtype=np.float64) + 0.5) * (in_size / out_size)
        - 0.5
    )


def resize_nearest(image, out_h, out_w):
    image = np.asarray(image, dtype=np.float64)
    in_h, in_w = image.shape[:2]
    ys = np.clip(
        np.rint(_sample_coords(in_h, out_h)).astype(np.int64),
        0,
        in_h - 1,
    )
    xs = np.clip(
        np.rint(_sample_coords(in_w, out_w)).astype(np.int64),
        0,
        in_w - 1,
    )
    return image[ys[:, None], xs[None, :]]


def resize_bilinear(image, out_h, out_w):
    image = np.asarray(image, dtype=np.float64)
    in_h, in_w = image.shape[:2]

    y = _sample_coords(in_h, out_h)
    x = _sample_coords(in_w, out_w)

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = (y - y0).reshape(-1, 1)
    wx = (x - x0).reshape(1, -1)

    y0c, y1c = np.clip(y0, 0, in_h - 1), np.clip(y1, 0, in_h - 1)
    x0c, x1c = np.clip(x0, 0, in_w - 1), np.clip(x1, 0, in_w - 1)

    if image.ndim == 3:
        wy = wy[..., None]
        wx = wx[..., None]

    top_left = image[y0c[:, None], x0c[None, :]]
    top_right = image[y0c[:, None], x1c[None, :]]
    bottom_left = image[y1c[:, None], x0c[None, :]]
    bottom_right = image[y1c[:, None], x1c[None, :]]

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    return top * (1.0 - wy) + bottom * wy


def antialias_sigma(in_size, out_size):
    decimation = in_size / float(out_size)
    if decimation <= 1.0:
        return 0.0
    return (decimation - 1.0) / 2.0


def resize(
    image,
    out_h,
    out_w,
    method="bilinear",
    antialias=True,
    pad_mode="reflect",
):
    image = np.asarray(image, dtype=np.float64)
    in_h, in_w = image.shape[:2]

    work = image
    if antialias:
        sigma_y = antialias_sigma(in_h, out_h)
        sigma_x = antialias_sigma(in_w, out_w)
        if sigma_y > 0.0 or sigma_x > 0.0:
            work = convolve_separable(
                work,
                gaussian_kernel1d(sigma_x),
                gaussian_kernel1d(sigma_y),
                pad_mode,
            )

    if method == "nearest":
        return resize_nearest(work, out_h, out_w)
    return resize_bilinear(work, out_h, out_w)


def add_gaussian_noise(image, sigma=0.05, seed=None):
    image = np.asarray(image, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return np.clip(image + rng.normal(0.0, sigma, image.shape), 0.0, 1.0)


def add_salt_pepper(image, amount=0.05, salt_ratio=0.5, seed=None):
    image = np.asarray(image, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = image.copy()
    draw = rng.random(image.shape[:2])
    salt = draw < (amount * salt_ratio)
    pepper = (draw >= (amount * salt_ratio)) & (draw < amount)
    out[salt] = 1.0
    out[pepper] = 0.0
    return out


def _median_plane(plane, size, pad_mode="reflect"):
    size = max(1, int(size))
    if size == 1:
        return plane
    radius = size // 2
    np_mode = PAD_MODES.get(pad_mode, "reflect")
    if np_mode == "constant":
        padded = np.pad(
            plane,
            radius,
            mode="constant",
            constant_values=0.0,
        )
    else:
        padded = np.pad(plane, radius, mode=np_mode)
    windows = sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-2, -1))


def median_filter2d(image, size=3, pad_mode="reflect"):
    return _per_channel(
        _median_plane,
        np.asarray(image, dtype=np.float64),
        size,
        pad_mode,
    )


def mean_filter2d(image, size=3, pad_mode="reflect"):
    n = max(1, int(size))
    ones = np.ones(n, dtype=np.float64) / n
    return convolve_separable(image, ones, ones, pad_mode)


def gaussian_blur(image, sigma=1.0, pad_mode="reflect"):
    k1d = gaussian_kernel1d(sigma)
    return convolve_separable(image, k1d, k1d, pad_mode)


def mse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def psnr(a, b, peak=1.0):
    error = mse(a, b)
    if error <= 1e-20:
        return float("inf")
    return float(10.0 * np.log10((peak ** 2) / error))


def ssim(a, b, sigma=1.5, peak=1.0):
    a = to_gray(np.asarray(a, dtype=np.float64))
    b = to_gray(np.asarray(b, dtype=np.float64))

    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2
    k1d = gaussian_kernel1d(sigma)

    def blur(plane):
        return convolve_separable(plane, k1d, k1d, "reflect")

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b

    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(numerator / denominator))


def difference_map(a, b, gain=1.0):
    diff = np.abs(
        np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    )
    if gain == "auto":
        peak = diff.max()
        gain = 1.0 if peak <= 1e-12 else 1.0 / peak
    return np.clip(diff * float(gain), 0.0, 1.0)
