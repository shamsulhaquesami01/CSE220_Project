"""Tests for the Spectral Rescue periodic-interference experiment."""

import numpy as np
from django.test import SimpleTestCase

from . import spectral_rescue as sr
from .operations import OPERATIONS


class SpectralRescueTests(SimpleTestCase):
    def setUp(self):
        h = w = 128
        y, x = np.mgrid[:h, :w]
        image = (
            0.35
            + 0.16 * np.sin(2 * np.pi * 2 * x / w)
            + 0.12 * np.sin(2 * np.pi * 3 * y / h)
            + 0.20
            * np.exp(
                -((x - 82) ** 2 + (y - 52) ** 2)
                / (2 * 15 ** 2)
            )
        )
        self.image = np.clip(image, 0.0, 1.0)

    def test_controlled_periodic_peak_is_detected_near_expected_bin(self):
        corrupted = sr.add_periodic_interference(
            self.image,
            pattern="vertical",
            frequency=20,
            amplitude=0.16,
            angle=0,
        )
        pairs, _ = sr.detect_spectral_peak_pairs(
            corrupted,
            sensitivity=6,
            ignore_center_fraction=0.04,
            max_pairs=4,
        )
        self.assertTrue(pairs)
        cy, cx = np.array(self.image.shape) // 2
        offsets = [
            (abs(y - cy), abs(x - cx))
            for y, x, _ in pairs
        ]
        self.assertTrue(
            any(
                dy <= 1 and abs(dx - 20) <= 1
                for dy, dx in offsets
            )
        )

    def test_notch_mask_is_symmetric_and_bounded(self):
        shape = (80, 96)
        pair = (21, 30, 1.0, "manual")
        mask = sr.build_notch_mask(
            shape, [pair], radius=4, softness=1
        )
        cy, cx = shape[0] // 2, shape[1] // 2
        my = (2 * cy - pair[0]) % shape[0]
        mx = (2 * cx - pair[1]) % shape[1]

        self.assertEqual(mask.shape, shape)
        self.assertGreaterEqual(mask.min(), 0.0)
        self.assertLessEqual(mask.max(), 1.0)
        self.assertAlmostEqual(
            mask[pair[0], pair[1]],
            mask[my, mx],
            places=12,
        )
        self.assertLess(mask[pair[0], pair[1]], 1e-8)

    def test_automatic_grid_removal_improves_psnr(self):
        corrupted = sr.add_periodic_interference(
            self.image,
            pattern="grid",
            frequency=20,
            amplitude=0.16,
            angle=25,
        )
        pairs, _ = sr.detect_spectral_peak_pairs(
            corrupted,
            sensitivity=6,
            ignore_center_fraction=0.04,
            max_pairs=5,
        )
        merged = sr._merge_pairs(
            pairs, [], self.image.shape
        )
        mask = sr.build_notch_mask(
            self.image.shape,
            merged,
            radius=3.5,
            softness=1,
        )
        restored = sr.apply_frequency_mask(
            corrupted, mask
        )

        self.assertGreater(
            sr.dsp.psnr(self.image, restored),
            sr.dsp.psnr(self.image, corrupted) + 3.0,
        )

    def test_manual_point_is_added_with_conjugate_partner(self):
        manual = sr._manual_pairs(
            [{"x": 0.25, "y": 0.30}],
            (100, 120),
        )
        merged = sr._merge_pairs(
            [], manual, (100, 120)
        )
        self.assertEqual(len(merged), 1)

        mask = sr.build_notch_mask(
            (100, 120),
            merged,
            radius=3,
            softness=1,
        )
        y, x, _, _ = merged[0]
        cy, cx = 50, 60
        my = (2 * cy - y) % 100
        mx = (2 * cx - x) % 120
        self.assertLess(mask[y, x], 1e-8)
        self.assertLess(mask[my, mx], 1e-8)

    def test_uploaded_mode_has_no_fake_reference_metrics(self):
        rgb = np.repeat(
            self.image[..., None], 3, axis=2
        )
        result = OPERATIONS[
            "spectral_rescue"
        ].handler(
            rgb,
            {
                "input_mode": "uploaded",
                "auto_detect": False,
                "manual_peaks": [
                    {"x": 0.2, "y": 0.4}
                ],
            },
        )
        labels = [
            metric.label for metric in result.metrics
        ]
        self.assertFalse(
            any("PSNR" in label for label in labels)
        )
        self.assertFalse(
            any("SSIM" in label for label in labels)
        )
        keys = {
            panel.key for panel in result.panels
        }
        self.assertIn("spectrum", keys)
        self.assertIn("mask", keys)
        self.assertIn("restored", keys)

    def test_operation_is_registered_and_finite(self):
        self.assertIn(
            "spectral_rescue", OPERATIONS
        )
        rgb = np.repeat(
            self.image[..., None], 3, axis=2
        )
        result = OPERATIONS[
            "spectral_rescue"
        ].handler(
            rgb,
            {
                "input_mode": "simulate",
                "pattern": "mesh",
                "pattern_frequency": 20,
                "pattern_amplitude": 0.12,
                "sensitivity": 6,
            },
        )
        self.assertGreaterEqual(
            len(result.panels), 6
        )
        self.assertGreater(
            len(result.metrics), 0
        )
        for panel in result.panels:
            self.assertEqual(
                panel.image.ndim, 3
            )
            self.assertTrue(
                np.isfinite(panel.image).all()
            )
