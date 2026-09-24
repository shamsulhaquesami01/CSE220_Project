"""Correctness tests for Fourier-Mellin Spectral Match."""

import io

import numpy as np
from PIL import Image
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase

from . import spectral_match as sm
from .operations import OPERATIONS


def structured_image(size=192):
    y, x = np.mgrid[:size, :size]
    image = np.zeros((size, size), dtype=np.float64)
    image += 0.82 * np.exp(-((x - 42) ** 2 + (y - 58) ** 2) / (2 * 9 ** 2))
    image += 0.64 * np.exp(-((x - 142) ** 2 + (y - 38) ** 2) / (2 * 16 ** 2))
    image += 0.72 * np.exp(-((x - 128) ** 2 + (y - 137) ** 2) / (2 * 21 ** 2))
    image[96:126, 24:74] += 0.56
    image[151:157, 52:164] += 0.90
    image[28:76, 164:170] += 0.76
    for xx in range(25, 148):
        yy = int(0.52 * xx + 40)
        if 2 <= yy < size - 2:
            image[yy - 2 : yy + 3, xx] += 0.42
    image = np.clip(image, 0.0, 1.0)
    return np.repeat(image[..., None], 3, axis=2)


def angle_error(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


class SpectralMatchCoreTests(SimpleTestCase):
    def test_phase_correlation_recovers_fractional_translation(self):
        reference = structured_image(128)[..., 0]
        query = sm.warp_similarity(reference, tx=7.4, ty=-5.7)
        result = sm.phase_correlation(reference, query, subpixel=True)

        # phase_correlation returns the shift that must be applied to query.
        self.assertAlmostEqual(result["shift_x"], -7.4, delta=0.6)
        self.assertAlmostEqual(result["shift_y"], 5.7, delta=0.6)
        self.assertGreater(result["psr"], 8.0)

    def test_fourier_mellin_recovers_rotation_scale_and_translation(self):
        source = structured_image()
        reference, _, reference_mask, _ = sm.prepare_pair(source)
        side = reference.shape[0]

        truth_angle = 37.0
        truth_scale = 0.76
        truth_tx = 0.08 * side
        truth_ty = -0.06 * side
        query = sm.warp_similarity(
            reference,
            angle_deg=truth_angle,
            scale=truth_scale,
            tx=truth_tx,
            ty=truth_ty,
        )
        query_mask = sm.warp_similarity(
            reference_mask,
            angle_deg=truth_angle,
            scale=truth_scale,
            tx=truth_tx,
            ty=truth_ty,
        )

        result = sm.register_similarity(
            reference,
            query,
            reference_mask,
            query_mask,
            use_hann=True,
            subpixel=True,
        )

        self.assertLess(angle_error(result["angle"], truth_angle), 1.0)
        self.assertAlmostEqual(result["scale"], truth_scale, delta=0.025)
        self.assertLess(
            np.hypot(result["tx"] - truth_tx, result["ty"] - truth_ty),
            2.5,
        )
        self.assertGreater(result["ncc"], 0.94)
        self.assertGreater(result["translation_phase"]["psr"], 8.0)

    def test_180_degree_ambiguity_is_resolved_spatially(self):
        source = structured_image()
        reference, _, reference_mask, _ = sm.prepare_pair(source)
        side = reference.shape[0]
        truth_angle = 151.0
        truth_scale = 1.12
        truth_tx = -0.05 * side
        truth_ty = 0.04 * side

        query = sm.warp_similarity(
            reference,
            angle_deg=truth_angle,
            scale=truth_scale,
            tx=truth_tx,
            ty=truth_ty,
        )
        query_mask = sm.warp_similarity(
            reference_mask,
            angle_deg=truth_angle,
            scale=truth_scale,
            tx=truth_tx,
            ty=truth_ty,
        )
        result = sm.register_similarity(
            reference,
            query,
            reference_mask,
            query_mask,
            use_hann=True,
            subpixel=True,
        )

        self.assertLess(angle_error(result["angle"], truth_angle), 1.5)
        self.assertAlmostEqual(result["scale"], truth_scale, delta=0.035)
        self.assertGreater(result["ncc"], 0.90)

    def test_controlled_operation_exposes_full_visual_pipeline(self):
        result = OPERATIONS["spectral_match"].handler(
            structured_image(144),
            {
                "input_mode": "controlled",
                "rotation": -28,
                "scale": 0.82,
                "shift_x_fraction": 0.08,
                "shift_y_fraction": -0.05,
                "use_hann": True,
                "subpixel": True,
            },
        )
        keys = {panel.key for panel in result.panels}
        expected = {
            "reference",
            "query",
            "reference_fft",
            "query_fft",
            "reference_log_polar",
            "query_log_polar",
            "rotation_scale_correlation",
            "translation_correlation",
            "aligned",
            "overlay",
            "checkerboard",
            "difference",
        }
        self.assertTrue(expected.issubset(keys))
        aligned = next(panel for panel in result.panels if panel.key == "aligned")
        self.assertIsNotNone(aligned.download_image)
        labels = {metric.label for metric in result.metrics}
        self.assertIn("Rotation error", labels)
        self.assertIn("Scale error", labels)
        self.assertIn("Translation error", labels)

    def test_real_mode_uses_second_image_and_has_no_ground_truth_metrics(self):
        reference = structured_image(144)
        query = sm.warp_similarity(reference, angle_deg=24, scale=0.88, tx=9, ty=-7)
        result = OPERATIONS["spectral_match"].handler(
            reference,
            {
                "input_mode": "real",
                "_query_image": query,
                "use_hann": True,
                "subpixel": True,
            },
        )
        labels = {metric.label for metric in result.metrics}
        self.assertNotIn("True rotation", labels)
        self.assertNotIn("Rotation error", labels)
        self.assertIn("Estimated rotation", labels)
        self.assertIn("Aligned correlation", labels)

    def test_real_mode_requires_query(self):
        with self.assertRaises(ValueError):
            OPERATIONS["spectral_match"].handler(
                structured_image(96),
                {"input_mode": "real"},
            )


class SpectralMatchApiTests(SimpleTestCase):
    @staticmethod
    def _upload(client, image, name):
        array = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        response = client.post(
            "/api/upload/",
            {
                "image": SimpleUploadedFile(
                    name,
                    buffer.getvalue(),
                    content_type="image/png",
                )
            },
        )
        return response

    def test_two_uploads_flow_through_process_endpoint(self):
        reference = structured_image(128)
        query = sm.warp_similarity(reference, angle_deg=20, scale=0.9, tx=8, ty=-6)

        reference_upload = self._upload(self.client, reference, "reference.png")
        query_upload = self._upload(self.client, query, "query.png")
        self.assertEqual(reference_upload.status_code, 200)
        self.assertEqual(query_upload.status_code, 200)

        response = self.client.post(
            "/api/process/",
            {
                "image_id": reference_upload.json()["image_id"],
                "op": "spectral_match",
                "params": {
                    "input_mode": "real",
                    "query_image_id": query_upload.json()["image_id"],
                    "use_hann": True,
                    "subpixel": True,
                },
            },
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        keys = {panel["key"] for panel in payload["panels"]}
        self.assertIn("aligned", keys)
        self.assertIn("overlay", keys)
        self.assertIn("checkerboard", keys)
        self.assertIn("difference", keys)
        aligned = next(panel for panel in payload["panels"] if panel["key"] == "aligned")
        self.assertTrue(aligned["download_url"])
