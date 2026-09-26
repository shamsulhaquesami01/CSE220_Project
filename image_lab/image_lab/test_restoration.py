"""Tests for the frequency-domain motion restoration experiment."""

import numpy as np
from django.test import SimpleTestCase

from . import dsp_utils as dsp
from .operations import OPERATIONS


class MotionRestorationTests(SimpleTestCase):
    def test_motion_psf_is_normalised_and_directional(self):
        horizontal = dsp.motion_psf(21, 0)
        diagonal = dsp.motion_psf(21, 45)
        self.assertAlmostEqual(horizontal.sum(), 1.0, places=12)
        self.assertAlmostEqual(diagonal.sum(), 1.0, places=12)
        self.assertGreater(np.count_nonzero(horizontal), 1)
        self.assertFalse(np.allclose(horizontal, diagonal))

    def test_psf_to_otf_has_unit_dc_gain(self):
        psf = dsp.motion_psf(15, 27)
        h = dsp.psf_to_otf(psf, (64, 80))
        self.assertEqual(h.shape, (64, 80))
        self.assertAlmostEqual(abs(h[0, 0]), 1.0, places=12)

    def test_identity_motion_reconstructs_without_noise(self):
        image = np.random.default_rng(4).random((32, 40, 3))
        blurred, degraded, inverse, wiener, _ = dsp.motion_deblur_experiment(
            image,
            length=1,
            angle=0,
            noise_sigma=0,
            wiener_k=1e-10,
            inverse_floor=1e-8,
            seed=0,
        )
        np.testing.assert_allclose(blurred, image, atol=1e-10)
        np.testing.assert_allclose(degraded, image, atol=1e-10)
        np.testing.assert_allclose(inverse, image, atol=1e-10)
        np.testing.assert_allclose(wiener, image, atol=1e-8)

    def test_wiener_is_stable_with_motion_blur_and_noise(self):
        # Synthetic image with edges and smooth areas.
        y, x = np.mgrid[:64, :64]
        image = ((x // 8 + y // 8) % 2).astype(np.float64)
        image = 0.15 + 0.7 * image

        _, degraded, inverse, wiener, _ = dsp.motion_deblur_experiment(
            image,
            length=17,
            angle=23,
            noise_sigma=0.012,
            wiener_k=2e-3,
            inverse_floor=1e-3,
            seed=7,
        )

        self.assertTrue(np.isfinite(inverse).all())
        self.assertTrue(np.isfinite(wiener).all())
        self.assertLessEqual(wiener.min(), 1.0)
        self.assertGreaterEqual(wiener.max(), 0.0)
        # Wiener output should improve the degraded image.
        self.assertGreater(dsp.psnr(image, wiener), dsp.psnr(image, degraded))

    def test_uploaded_observation_is_not_blurred_again(self):
        y, x = np.mgrid[:48, :56]
        observed = ((x // 7 + y // 7) % 2).astype(np.float64)
        inverse, wiener, _ = dsp.motion_deblur_observation(
            observed,
            length=1,
            angle=0,
            wiener_k=1e-10,
            inverse_floor=1e-8,
        )
        np.testing.assert_allclose(inverse, observed, atol=1e-10)
        np.testing.assert_allclose(wiener, observed, atol=1e-8)

    def test_uploaded_mode_restores_a_generated_degraded_signal(self):
        y, x = np.mgrid[:48, :56]
        image = (0.2 + 0.6 * ((x // 7 + y // 7) % 2)).astype(np.float64)
        _, degraded, _, _, _ = dsp.motion_deblur_experiment(
            image,
            length=9,
            angle=14,
            noise_sigma=0,
            wiener_k=2e-3,
            inverse_floor=1e-3,
        )
        inverse, wiener, _ = dsp.motion_deblur_observation(
            degraded,
            length=9,
            angle=14,
            wiener_k=2e-3,
            inverse_floor=1e-3,
        )
        self.assertEqual(inverse.shape, image.shape)
        self.assertEqual(wiener.shape, image.shape)
        self.assertTrue(np.isfinite(inverse).all())
        self.assertTrue(np.isfinite(wiener).all())

    def test_uploaded_mode_has_no_fake_reference_metrics(self):
        image = np.random.default_rng(8).random((32, 40, 3))
        result = OPERATIONS["deblur"].handler(
            image,
            {"input_mode": "uploaded", "motion_length": 9, "motion_angle": 12},
        )
        self.assertEqual(result.panels[0].key, "observed")
        self.assertNotIn("degraded", {panel.key for panel in result.panels})
        self.assertFalse(any("PSNR" in metric.label for metric in result.metrics))
        self.assertFalse(any("SSIM" in metric.label for metric in result.metrics))

    def test_deblur_operation_is_registered(self):
        self.assertIn("deblur", OPERATIONS)
        image = np.random.default_rng(9).random((32, 32, 3))
        result = OPERATIONS["deblur"].handler(
            image,
            {
                "motion_length": 11,
                "motion_angle": 15,
                "noise_sigma": 0.005,
                "wiener_log10": -3,
                "seed": 2,
            },
        )
        self.assertGreaterEqual(len(result.panels), 5)
        self.assertGreater(len(result.metrics), 0)
        for panel in result.panels:
            self.assertEqual(panel.image.ndim, 3)
            self.assertTrue(np.isfinite(panel.image).all())
