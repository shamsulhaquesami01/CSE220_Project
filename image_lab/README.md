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
