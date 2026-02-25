"""Tests for s2sharp.convolution module."""

import numpy as np
import pytest

from s2sharp.convolution import (
    conv_cm,
    create_conv_kernel,
    create_conv_kernel_subspace,
    create_diff_kernels,
)
from s2sharp.utils import conv2im, conv2mat


class TestCreateConvKernel:
    """Test blur kernel creation."""

    def test_output_shape(self):
        nl, nc, L = 60, 60, 12
        d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
        sdf = d * np.sqrt(-2 * np.log(np.array([.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23])) / np.pi ** 2)
        sdf[d == 1] = 0
        FBM = create_conv_kernel(sdf, d, nl, nc, L, 12, 12)
        assert FBM.shape == (nl, nc, L)

    def test_identity_for_10m_bands(self):
        """Bands with d=1 should have identity kernel (delta function in FFT)."""
        nl, nc, L = 60, 60, 12
        d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
        sdf = np.zeros(L)
        FBM = create_conv_kernel(sdf, d, nl, nc, L, 12, 12)
        # For d==1 bands, FBM should be all ones (FFT of delta)
        for i in np.where(d == 1)[0]:
            np.testing.assert_allclose(FBM[:, :, i], np.ones((nl, nc)), atol=1e-10)


class TestCreateConvKernelSubspace:
    """Test complementary blur kernel creation."""

    def test_output_shape(self):
        nl, nc, L = 60, 60, 12
        sdf = np.array([2.0, 0, 0, 0, 0.8, 0.8, 0.8, 0, 0.8, 2.0, 0.8, 0.8])
        FBM2 = create_conv_kernel_subspace(sdf, nl, nc, L, 12, 12)
        assert FBM2.shape == (nl, nc, L)

    def test_max_blur_band_is_identity(self):
        """Band with maximum blur should get identity (no additional blur needed)."""
        nl, nc, L = 60, 60, 3
        sdf = np.array([2.0, 1.0, 0.5])
        FBM2 = create_conv_kernel_subspace(sdf, nl, nc, L, 12, 12)
        # Band 0 has max sdf, should be delta (all ones in FFT)
        np.testing.assert_allclose(FBM2[:, :, 0], np.ones((nl, nc)), atol=1e-10)


class TestConvCM:
    """Test circular convolution in matrix format."""

    def test_identity_convolution(self):
        """Convolving with delta kernel should return the input."""
        nl, nc, L = 10, 8, 3
        X = np.random.randn(L, nl * nc)
        # Delta kernel in FFT domain = all ones
        FKM = np.ones((nl, nc, L))
        result = conv_cm(X, FKM, nl)
        np.testing.assert_allclose(result, X, atol=1e-10)

    def test_output_shape(self):
        nl, nc, L = 10, 8, 4
        X = np.random.randn(L, nl * nc)
        FKM = np.ones((nl, nc, L), dtype=complex)
        result = conv_cm(X, FKM, nl)
        assert result.shape == (L, nl * nc)


class TestCreateDiffKernels:
    """Test finite difference kernel creation."""

    def test_output_shapes(self):
        nl, nc, r = 20, 30, 5
        FDH, FDV, FDHC, FDVC = create_diff_kernels(nl, nc, r)
        assert FDH.shape == (nl, nc, r)
        assert FDV.shape == (nl, nc, r)
        assert FDHC.shape == (nl, nc, r)
        assert FDVC.shape == (nl, nc, r)

    def test_conjugate_relationship(self):
        nl, nc, r = 20, 30, 5
        FDH, FDV, FDHC, FDVC = create_diff_kernels(nl, nc, r)
        np.testing.assert_allclose(FDHC, np.conj(FDH))
        np.testing.assert_allclose(FDVC, np.conj(FDV))

    def test_replicated_across_r(self):
        """All r slices should be identical."""
        nl, nc, r = 20, 30, 5
        FDH, FDV, _, _ = create_diff_kernels(nl, nc, r)
        for k in range(1, r):
            np.testing.assert_allclose(FDH[:, :, k], FDH[:, :, 0])
            np.testing.assert_allclose(FDV[:, :, k], FDV[:, :, 0])
