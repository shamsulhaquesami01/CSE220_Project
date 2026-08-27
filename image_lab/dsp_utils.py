from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Padding modes we expose in the UI, mapped to the equivalent np.pad mode.
PAD_MODES = {
    "reflect": "reflect",      # abc -> cba|abc|cba   (no edge duplication)
    "edge": "edge",            # abc -> aaa|abc|ccc   (clamp / replicate)
    "wrap": "wrap",            # abc -> abc|abc|abc   (circular convolution)
    "zero": "constant",        # abc -> 000|abc|000   (linear convolution)
}


# Small helpers


# Return the image as float64 in [0, 1], accepting uint8 or float input.
def to_float(image):
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.astype(np.float64) / 255.0
    return np.clip(array.astype(np.float64), 0.0, 1.0)

# Return the image as uint8 in [0, 255], clipping out-of-range values.
def to_uint8(image):
    return np.clip(np.asarray(image) * 255.0, 0.0, 255.0).round().astype(np.uint8)

# Apply a single-plane function to every colour channel and restack the result.
def _per_channel(func, image, *args, **kwargs):
    image = np.asarray(image)
    if image.ndim == 2:
        return func(image, *args, **kwargs)
    planes = [func(image[..., c], *args, **kwargs) for c in range(image.shape[-1])]
    return np.stack(planes, axis=-1)

# Return the luminance plane of an image using the ITU-R BT.601 weights.
def to_gray(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    return image[..., :3] @ np.array([0.299, 0.587, 0.114])


# 2D convolution


# Return the (top, bottom, left, right) pad widths that keep the output size.
def _anchor_padding(kh, kw):
    top = kh // 2
    left = kw // 2
    return top, kh - 1 - top, left, kw - 1 - left

# Convolve one 2D plane with a 2D kernel, preserving the input shape.
def _convolve2d_plane(plane, kernel, pad_mode="reflect"):
    kh, kw = kernel.shape
    top, bottom, left, right = _anchor_padding(kh, kw)
    
    # Reverse the kernel on both axes so the correlation below equals convolution.
    flipped = kernel[::-1, ::-1]
    
    np_mode = PAD_MODES.get(pad_mode, "reflect")
    if np_mode == "constant":
        padded = np.pad(plane, ((top, bottom), (left, right)), mode="constant", constant_values=0.0)
    else:
        padded = np.pad(plane, ((top, bottom), (left, right)), mode=np_mode)
        
    # sliding_window_view is a stride trick: it produces an (H, W, kh, kw) view
    # of every kh-by-kw neighbourhood without copying the underlying data.
    windows = sliding_window_view(padded, (kh, kw))
    
    # Contract each window against the flipped kernel -> one output sample.
    return np.einsum("mnuv,uv->mn", windows, flipped, optimize=True)

# Convolve an image (grayscale or colour) with a 2D kernel.
def convolve2d(image, kernel, pad_mode="reflect"):
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array")
    if kernel.size == 0:
        raise ValueError("Kernel must not be empty")
    return _per_channel(_convolve2d_plane, np.asarray(image, dtype=np.float64), kernel, pad_mode)

# Convolve one 2D plane with a separable kernel given as two 1D factors.
def _convolve_separable_plane(plane, kx, ky, pad_mode="reflect"):
    out = plane
    if kx.size > 1 or not np.isclose(kx[0], 1.0):
        out = _convolve2d_plane(out, kx.reshape(1, -1), pad_mode)
    if ky.size > 1 or not np.isclose(ky[0], 1.0):
        out = _convolve2d_plane(out, ky.reshape(-1, 1), pad_mode)
    return out

# Convolve an image with a separable kernel defined by 1D factors kx and ky.
def convolve_separable(image, kx, ky, pad_mode="reflect"):
    kx = np.asarray(kx, dtype=np.float64).ravel()
    ky = np.asarray(ky, dtype=np.float64).ravel()
    return _per_channel(_convolve_separable_plane, np.asarray(image, dtype=np.float64), kx, ky, pad_mode)


# Kernel construction


# Return a 1D Gaussian kernel sampled at integer offsets and normalised to sum 1.
def gaussian_kernel1d(sigma, radius=None):
    if sigma <= 0:
        return np.array([1.0])
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()

# Return a square 2D Gaussian kernel built as the outer product of 1D factors.
def gaussian_kernel2d(sigma, size=None):
    radius = None if size is None else max(1, int(size) // 2)
    k1d = gaussian_kernel1d(sigma, radius)
    return np.outer(k1d, k1d)

# Return an n-by-n box (moving average) kernel.
def box_kernel(n):
    n = max(1, int(n))
    return np.full((n, n), 1.0 / (n * n), dtype=np.float64)

# Scale a kernel so its coefficients sum to 1, leaving zero-sum kernels alone.
def normalize_kernel(kernel):
    kernel = np.asarray(kernel, dtype=np.float64)
    total = kernel.sum()
    if np.isclose(total, 0.0):
        return kernel
    return kernel / total

KERNEL_PRESETS = {
    "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    "box_blur_3": [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    "gaussian_3": [[1 / 16, 2 / 16, 1 / 16], [2 / 16, 4 / 16, 2 / 16], [1 / 16, 2 / 16, 1 / 16]],
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "sharpen_strong": [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
    "laplacian": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    "sobel_x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "sobel_y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
}

# Return the unsharp-mask result: original + amount * (original - blurred).
def unsharp_mask(image, sigma=1.0, amount=1.0, pad_mode="reflect"):
    k1d = gaussian_kernel1d(sigma)
    blurred = convolve_separable(image, k1d, k1d, pad_mode)
    detail = np.asarray(image, dtype=np.float64) - blurred
    return np.asarray(image, dtype=np.float64) + amount * detail

# ---------------------------------------------------------------------------
# Resampling: nearest neighbour, bilinear, and anti-aliased downsampling
# ---------------------------------------------------------------------------

# All resampling here uses the *half-pixel centre* convention. Pixel k of an
# N-wide row is treated as covering [k, k+1) with its centre at k + 0.5. To find
# where output pixel `d` lands in the input we match normalised centres:
#     (d + 0.5) / out = (s + 0.5) / in
#     s = (d + 0.5) * (in / out) - 0.5
# This is the convention that keeps an image geometrically centred after a
# resize. The naive alternative, s = d * (in / out), shifts the picture by half
# a pixel and is a classic source of "my resized image drifted" bugs.

# Return the source coordinates that each output sample reads from.
def _sample_coords(in_size, out_size):
    return (np.arange(out_size, dtype=np.float64) + 0.5) * (in_size / out_size) - 0.5

# Resize an image by picking the single closest input pixel for each output pixel.
def resize_nearest(image, out_h, out_w):
    image = np.asarray(image, dtype=np.float64)
    in_h, in_w = image.shape[:2]
    ys = np.clip(np.rint(_sample_coords(in_h, out_h)).astype(np.int64), 0, in_h - 1)
    xs = np.clip(np.rint(_sample_coords(in_w, out_w)).astype(np.int64), 0, in_w - 1)
    # Advanced indexing broadcasts to (out_h, out_w) and carries channels along.
    return image[ys[:, None], xs[None, :]]

# Resize an image by bilinear interpolation of the four surrounding pixels.
def resize_bilinear(image, out_h, out_w):
    image = np.asarray(image, dtype=np.float64)
    in_h, in_w = image.shape[:2]
    y = _sample_coords(in_h, out_h)
    x = _sample_coords(in_w, out_w)
    
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    
    # Fractional distance from the top-left neighbour, i.e. the blend weights.
    wy = (y - y0).reshape(-1, 1)
    wx = (x - x0).reshape(1, -1)
    
    # Clamp *after* computing the weights so edge pixels extend rather than wrap.
    y0c, y1c = np.clip(y0, 0, in_h - 1), np.clip(y1, 0, in_h - 1)
    x0c, x1c = np.clip(x0, 0, in_w - 1), np.clip(x1, 0, in_w - 1)
    
    if image.ndim == 3:
        wy = wy[..., None]
        wx = wx[..., None]
        
    top_left = image[y0c[:, None], x0c[None, :]]
    top_right = image[y0c[:, None], x1c[None, :]]
    bottom_left = image[y1c[:, None], x0c[None, :]]
    bottom_right = image[y1c[:, None], x1c[None, :]]
    
    # Interpolate along x first, then blend the two rows along y.
    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    return top * (1.0 - wy) + bottom * wy

# Return the Gaussian sigma needed to band-limit before a given downsample.
def antialias_sigma(in_size, out_size):
    decimation = in_size / float(out_size)
    if decimation <= 1.0:
        return 0.0
    return (decimation - 1.0) / 2.0

# Resize an image, optionally low-pass filtering first to suppress aliasing.
def resize(image, out_h, out_w, method="bilinear", antialias=True, pad_mode="reflect"):
    image = np.asarray(image, dtype=np.float64)
    in_h, in_w = image.shape[:2]
    work = image
    
    if antialias:
        sigma_y = antialias_sigma(in_h, out_h)
        sigma_x = antialias_sigma(in_w, out_w)
        if sigma_y > 0.0 or sigma_x > 0.0:
            # Separate sigmas so non-uniform scaling is filtered correctly on
            # each axis. A zero sigma yields the length-1 identity kernel.
            work = convolve_separable(
                work,
                gaussian_kernel1d(sigma_x),
                gaussian_kernel1d(sigma_y),
                pad_mode,
            )
            
    if method == "nearest":
        return resize_nearest(work, out_h, out_w)
    return resize_bilinear(work, out_h, out_w)