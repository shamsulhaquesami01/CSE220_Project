import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings

from . import dsp_utils as dsp


class ResultDownloadTests(SimpleTestCase):
    def test_resample_downloads_keep_target_pixels_and_preview_keeps_source_size(self):
        with tempfile.TemporaryDirectory() as media:
            with override_settings(MEDIA_ROOT=media):
                source = np.random.default_rng(0).integers(
                    0, 256, (24, 36, 3), dtype=np.uint8
                )
                buffer = io.BytesIO()
                Image.fromarray(source).save(buffer, format="PNG")
                upload = self.client.post("/api/upload/", {
                    "image": SimpleUploadedFile(
                        "source.png", buffer.getvalue(), content_type="image/png"
                    ),
                })
                self.assertEqual(upload.status_code, 200)
                image_id = upload.json()["image_id"]

                def read_png(url):
                    path = Path(media) / url.removeprefix(settings.MEDIA_URL)
                    with Image.open(path) as handle:
                        return np.array(handle)

                for scale in (2.0, 0.5, 1.0):
                    with self.subTest(scale=scale):
                        params = {"scale": scale, "method": "bilinear"}
                        # Repeat to exercise saved result reuse as well.
                        for _ in range(2):
                            response = self.client.post(
                                "/api/process/",
                                {"image_id": image_id, "op": "resample", "params": params},
                                content_type="application/json",
                            )
                            self.assertEqual(response.status_code, 200)
                            for panel in response.json()["panels"]:
                                preview = read_png(panel["url"])
                                self.assertEqual(preview.shape, source.shape)
                                exported = read_png(panel["download_url"])
                                self.assertEqual(
                                    exported.shape[:2],
                                    (panel["download_height"], panel["download_width"]),
                                )
                                if panel["key"] in ("aa", "no_aa"):
                                    height, width = int(24 * scale), int(36 * scale)
                                    expected = dsp.to_uint8(dsp.resize(
                                        dsp.to_float(source), height, width,
                                        antialias=panel["key"] == "aa",
                                    ))
                                    np.testing.assert_array_equal(exported, expected)
                                    self.assertNotEqual(panel["url"], panel["download_url"])
                                else:
                                    self.assertEqual(panel["url"], panel["download_url"])

                response = self.client.post(
                    "/api/process/",
                    {"image_id": image_id, "op": "convolve", "params": {"kernel": [[1]]}},
                    content_type="application/json",
                )
                self.assertEqual(response.status_code, 200)
                for panel in response.json()["panels"]:
                    self.assertEqual(panel["url"], panel["download_url"])
                    self.assertEqual(read_png(panel["download_url"]).shape, source.shape)
