# Image Lab

A Django GUI for 2D DSP experiments: convolution filtering, resampling with and
without anti-aliasing, and noise injection / restoration.

Built for CSE 220 (Signals & Linear Systems). Every algorithm is implemented
from its defining equation in `image_lab/dsp_utils.py` using NumPy only — there
is no OpenCV, and `scipy.ndimage` is deliberately **not** a dependency.

---

## Running it

```bash
cd CSE220/image_lab
python -m pip install -r requirements.txt
python manage.py runserver
```


## mac 

cd image_lab
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 manage.py runserver


Open <http://127.0.0.1:8000/>. There is no database and no migration step.

```bash
python manage.py test image_lab     # 23 tests
python manage.py check
```

### Upload storage

Local uploads are normalised to PNG and saved in `media/uploads`; generated
panels are cached in `media/results`. On Vercel, both groups are stored in the
connected Vercel Blob store under `uploads/` and `results/`, so they survive
between serverless requests. Images larger than the configured 720-pixel edge
limit are resized during upload.

### Motion deblurring

“Deblur uploaded image” treats the upload as the already-blurred observation
and applies direct inverse and Wiener deconvolution using the selected motion
length and angle. It does not add another blur. This is non-blind restoration,
so the user supplies the estimated blur parameters; PSNR and SSIM are omitted
because no clean reference exists. “Simulate blur, then restore” preserves the
controlled teaching experiment and its reference-based quality metrics.

### Result image inspector

Click any result image (or focus it and press Enter) to inspect the downloadable
PNG. Comparison cards keep their existing preview dimensions. The inspector
starts at 1:1: one image pixel occupies one CSS pixel, subject to browser zoom
and display density. Fit to screen shrinks oversized images without enlarging
small ones. Use +/− to zoom, drag or scroll to pan, and Escape, the backdrop, or
× to close. Dimensions come from the loaded export, and the download link uses
the same URL and filename as the card.

Browser regression coverage (requires Node.js, Playwright, and Chrome): start
`python manage.py runserver 127.0.0.1:8765`, then run
`node browser_tests/inspector.cjs`. Set `TEST_URL` to use another local server.
The test uploads a generated PNG and checks all four operations, native export
sizes, zoom/pan/fit, keyboard focus, close controls, mobile layout, and load errors.


## Spectral Rescue: periodic obstruction removal

Spectral Rescue targets **repetitive visual interference**, not arbitrary opaque
foreground objects. It is designed for regular mesh/screen patterns, horizontal
or vertical scan lines, periodic stripes, regular banding, and similar
near-periodic contamination.

The pipeline is:

1. Convert the observed image to luminance and compute a centred 2-D FFT.
2. Display the log-magnitude spectrum.
3. Ignore a configurable low-frequency centre region.
4. Estimate the slowly varying spectral background.
5. Detect narrow local spectral anomalies with a robust MAD-based threshold.
6. Pair conjugate-symmetric peaks.
7. Merge automatic peaks with manually clicked spectrum locations.
8. Build smooth symmetric Gaussian notch-reject filters.
9. Apply the same frequency mask to each image channel.
10. Reconstruct with the inverse FFT.

The experiment has two modes:

- **Simulate + recover** adds controlled periodic interference to the uploaded
  clean image. This is the recommended first test because PSNR and SSIM can
  verify whether reconstruction actually improved the image.
- **Clean an uploaded obstruction** treats the upload as an already-corrupted
  real image. It intentionally does not report PSNR/SSIM because no clean
  reference exists.

### Spectral Rescue controls

- Interference preset: horizontal, vertical, diagonal, grid, multi-frequency mesh
- Pattern frequency, strength, and angle for controlled simulation
- Automatic peak detection on/off
- Detection sensitivity
- Ignore-centre radius
- Maximum automatic peak pairs
- Require conjugate-symmetric evidence
- Notch radius
- Notch softness
- Peak marker display
- Manual peak selection by clicking the Fourier-spectrum panel
- Clear manual peaks

Clicking one spectrum location automatically creates the conjugate-symmetric
partner. This keeps the filtered spectrum compatible with a real-valued inverse
FFT.

The result panels include the original/reference when available, corrupted
observation, annotated Fourier spectrum, notch mask, reconstructed image,
filtered spectrum, and the suppressed component.

### Recommended Spectral Rescue test

Start with any detailed photograph and use **Simulate + recover**:

- preset: Two-direction grid
- frequency: 24 cycles/image
- strength: 16%
- angle: 25 degrees
- auto detect: on
- sensitivity: 6/10
- centre radius: 5%
- max pairs: 5
- notch radius: 4 px
- notch softness: 1.0

Confirm that the bright symmetric peaks are marked, dark holes appear in the
notch mask, and PSNR/SSIM improve after reconstruction.

Then try uploaded-mode images containing fine screens, scanlines, regular
stripes, or sensor banding. For real images, tune sensitivity first, then notch
radius. If automation misses a visible Fourier peak, click it manually.

This method is **not expected to remove** people, branches, irregular fences,
large opaque bars, or arbitrary objects that completely hide scene content.
Those problems require additional observations, inpainting/ML, or multi-frame
computer-vision methods.
