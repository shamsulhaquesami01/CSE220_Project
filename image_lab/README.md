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
python manage.py test image_lab
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

### Spectral Match (Fourier-Mellin registration)

Spectral Match aligns two views of substantially the same planar image without
OpenCV features or machine learning. It estimates a similarity transform:

- in-plane rotation,
- uniform scale,
- horizontal translation,
- vertical translation.

The processing chain is:

1. Optional 2-D Hann window.
2. Centred 2-D FFT magnitude for the reference and query.
3. Log-polar resampling of both Fourier magnitudes.
4. Phase correlation in log-polar space to estimate rotation and scale.
5. Explicit 180-degree ambiguity resolution using spatial registration quality.
6. Undo estimated rotation/scale.
7. A second phase-correlation pass to estimate translation.
8. Optional parabolic subpixel peak refinement.
9. Registered output, overlay, checkerboard comparison, and difference map.

Two modes are available:

- **Controlled transform challenge**: one upload is transformed by known
  rotation, scale, and shifts. The estimator never receives those true values;
  they are shown afterward so rotation/scale/translation errors can be measured.
- **Match two uploaded images**: the first upload is the reference and a second
  upload becomes the query. This works best when both show the same flat
  scene/object and differ mainly by rotation, uniform zoom, and translation.

The result grid exposes both FFT magnitudes, both log-polar images, the
rotation/scale phase-correlation surface, the translation phase-correlation
surface, registered query, overlay, checkerboard, and difference image. The
registered image can be downloaded like any other panel.

Useful metrics include the estimated transform, phase-correlation PSR and peak
ratios, normalized aligned correlation, overlap fraction, overlap MSE, and
controlled-mode ground-truth errors.

Focused tests:

    python manage.py test image_lab.test_spectral_match

A good first controlled test is rotation +32 degrees, scale 0.78x, horizontal
shift +10%, vertical shift -7%, Hann enabled, and subpixel refinement enabled.

For real two-image tests, start with a screenshot, poster, document, map, book
cover, PCB, or other mostly planar target. Create the query by rotating,
uniformly resizing, shifting, or mildly cropping the same image. Strong
perspective changes, unrelated images, deforming objects, or substantially
different viewpoints are outside the similarity-transform model.
