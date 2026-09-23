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
