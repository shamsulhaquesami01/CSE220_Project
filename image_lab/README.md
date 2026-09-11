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

Open <http://127.0.0.1:8000/>. There is no database and no migration step.

```bash
python manage.py test image_lab     # 23 tests
python manage.py check
```