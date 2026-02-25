"""Tests for s2sharp.metrics module."""

import numpy as np
import pytest

from s2sharp.metrics import sam, ergas, sre, compute_ssim


class TestSAM:
    """Test Spectral Angle Mapper."""

    def test_identical_images(self):
        """SAM of identical images should be 0."""
        I = np.random.rand(10, 10, 5) + 0.1
        sam_val, sam_map = sam(I, I)
        assert sam_val == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_spectra(self):
        """SAM of orthogonal spectra should be 90 degrees."""
        I1 = np.zeros((1, 1, 2))
        I2 = np.zeros((1, 1, 2))
        I1[0, 0, 0] = 1.0
        I2[0, 0, 1] = 1.0
        sam_val, _ = sam(I1, I2)
        assert sam_val == pytest.approx(90.0, abs=1e-6)

    def test_sam_map_shape(self):
        M, N, B = 8, 6, 4
        I1 = np.random.rand(M, N, B) + 0.1
        I2 = np.random.rand(M, N, B) + 0.1
        _, sam_map = sam(I1, I2)
        assert sam_map.shape == (M, N)


class TestERGAS:
    """Test ERGAS metric."""

    def test_identical_images(self):
        """ERGAS of identical images should be 0."""
        I = np.random.rand(10, 10, 3) + 0.1
        val = ergas(I, I, 2)
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_positive(self):
        """ERGAS should be positive for different images."""
        I1 = np.random.rand(10, 10, 3) + 0.1
        I2 = I1 + 0.01 * np.random.randn(10, 10, 3)
        val = ergas(I1, I2, 2)
        assert val > 0


class TestSRE:
    """Test Signal-to-Reconstruction Error."""

    def test_known_snr(self):
        """SRE should match known SNR."""
        L, n = 3, 100
        X_true = np.ones((L, n))
        # Add noise with known power
        noise = 0.1 * np.ones((L, n))
        X_est = X_true + noise
        result = sre(X_true, X_est)
        # SRE = 10*log10(signal_power / error_power)
        # signal_power = 100, error_power = 100 * 0.01 = 1
        expected = 10 * np.log10(100 / 1.0)
        for i in range(L):
            assert result[i] == pytest.approx(expected, rel=1e-6)

    def test_identical_signals(self):
        """SRE of identical signals should be infinite (or very large)."""
        L, n = 2, 50
        X = np.random.rand(L, n) + 0.1
        result = sre(X, X)
        assert all(np.isinf(result) | (result > 100))


class TestSSIM:
    """Test per-band SSIM."""

    def test_identical_images(self):
        """SSIM of identical images should be 1."""
        I = np.random.rand(20, 20, 3)
        result = compute_ssim(I, I)
        for val in result:
            assert val == pytest.approx(1.0, abs=1e-6)

    def test_output_shape(self):
        B = 5
        I1 = np.random.rand(20, 20, B)
        I2 = np.random.rand(20, 20, B)
        result = compute_ssim(I1, I2)
        assert result.shape == (B,)
