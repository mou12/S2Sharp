"""Tests comparing Python S2Sharp against MATLAB reference values.

Run export_reference.m in MATLAB first to generate Data/matlab_reference.mat,
then run these tests with: pytest tests/test_matlab_comparison.py -v

These tests validate that each phase of the MATLAB→Python port produces
matching results.
"""

import os

import numpy as np
import pytest
import scipy.io

# Path to MATLAB reference data
REF_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'matlab_reference.mat')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Aviris_cell_3.mat')


def load_reference():
    """Load MATLAB reference values, skipping if file doesn't exist."""
    if not os.path.exists(REF_PATH):
        pytest.skip("MATLAB reference file not found. Run export_reference.m first.")
    return scipy.io.loadmat(REF_PATH, squeeze_me=True)


def load_dataset():
    """Load the Aviris dataset."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("Dataset not found at Data/Aviris_cell_3.mat")
    return scipy.io.loadmat(DATA_PATH, squeeze_me=True, simplify_cells=True)


@pytest.fixture(scope="module")
def ref():
    return load_reference()


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def setup(dataset, ref):
    """Set up parameters and run initialization, returning intermediate values."""
    from s2sharp.convolution import create_conv_kernel_subspace
    from s2sharp.initialization import initialize
    from s2sharp.preprocessing import compute_weights, normalize_data

    data = dataset
    Yim = data['Yim']
    bands_raw = [Yim[i].astype(np.float64) for i in range(len(Yim))]

    L = len(bands_raw)
    nl, nc = bands_raw[1].shape  # Band 2 (10m) defines dimensions

    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    mtf = np.array([.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23])
    sdf = d * np.sqrt(-2 * np.log(mtf) / np.pi ** 2)
    sdf[d == 1] = 0

    bands, av = normalize_data(bands_raw)

    dx, dy = 12, 12
    r = 8
    border = 2

    FBM2 = create_conv_kernel_subspace(sdf, nl, nc, L, dx, dy)
    Y, M, F = initialize(bands, sdf, nl, nc, L, dx, dy, d, border, r)

    n = nl * nc
    from s2sharp.utils import conv2mat
    Mask = conv2mat(M)

    W = compute_weights(Y, d, 1, nl)

    return {
        'FBM2': FBM2,
        'Y': Y, 'Mask': Mask, 'F': F,
        'W': W,
        'nl': nl, 'nc': nc, 'L': L, 'r': r,
        'sdf': sdf, 'd': d, 'av': av,
        'bands': bands, 'dx': dx, 'dy': dy, 'border': border,
    }


class TestPhase1Rounding:
    """Phase 1: Validate FBM2 matches MATLAB (rounding fix)."""

    def test_fbm2_matches(self, ref, setup):
        FBM2_py = setup['FBM2']
        FBM2_mat = ref['FBM2']
        np.testing.assert_allclose(FBM2_py, FBM2_mat, atol=1e-12,
                                   err_msg="FBM2 mismatch — rounding fix may be incorrect")


class TestPhase2Interpolation:
    """Phase 2: Validate Ylim and F_init match MATLAB (imresize fix)."""

    def test_ylim_matches(self, ref, dataset):
        from s2sharp.preprocessing import normalize_data
        from s2sharp.utils import matlab_imresize

        data = dataset
        Yim = data['Yim']
        bands_raw = [Yim[i].astype(np.float64) for i in range(len(Yim))]
        bands, _ = normalize_data(bands_raw)

        d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
        nl, nc = bands[1].shape
        L = len(bands)

        Ylim = np.zeros((nl, nc, L))
        for i in range(L):
            di = d[i]
            if di == 1:
                Ylim[:, :, i] = bands[i]
            else:
                Ylim[:, :, i] = matlab_imresize(bands[i], di)

        Ylim_mat = ref['Ylim']
        np.testing.assert_allclose(Ylim, Ylim_mat, atol=1e-8,
                                   err_msg="Ylim mismatch — imresize fix may be incorrect")

    def test_f_init_matches(self, ref, setup):
        F_py = setup['F']
        F_mat = ref['F_init']
        # SVD sign ambiguity: compare absolute values of columns
        for col in range(F_py.shape[1]):
            # Check if columns match up to sign
            if np.dot(F_py[:, col], F_mat[:, col]) < 0:
                F_py_col = -F_py[:, col]
            else:
                F_py_col = F_py[:, col]
            np.testing.assert_allclose(F_py_col, F_mat[:, col], atol=1e-6,
                                       err_msg=f"F_init column {col} mismatch")


class TestPhase4Weights:
    """Phase 4: Validate W matches MATLAB (boundary conditions fix)."""

    def test_weights_match(self, ref, setup):
        W_py = setup['W']
        W_mat = ref['W']
        np.testing.assert_allclose(W_py, W_mat, atol=1e-10,
                                   err_msg="W mismatch — boundary conditions fix may be incorrect")


class TestPhase3Optimizer:
    """Phase 3: Validate F after first F-step matches MATLAB (TrustRegions fix)."""

    def test_f_iter1_matches(self, ref, setup):
        from s2sharp.convolution import create_conv_kernel, create_diff_kernels
        from s2sharp.optimization import f_step
        from s2sharp.solvers import z_step

        nl, nc, L, r = setup['nl'], setup['nc'], setup['L'], setup['r']
        n = nl * nc
        Y, Mask, F = setup['Y'], setup['Mask'], setup['F']
        d, sdf = setup['d'], setup['sdf']

        FBM = create_conv_kernel(sdf, d, nl, nc, L, 12, 12)
        FDH, FDV, FDHC, FDVC = create_diff_kernels(nl, nc, r)
        W = setup['W']
        q = np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689])
        lam = 1.8998e-04

        Z = np.zeros((r, n))
        Z = z_step(Y, FBM, F, lam, nl, nc, Z, Mask, q, FDH, FDV, FDHC, FDVC, W)

        # Compare Z after first Z-step
        Z_mat = ref['Z_iter1']
        np.testing.assert_allclose(Z, Z_mat, rtol=1e-4, atol=1e-8,
                                   err_msg="Z_iter1 mismatch")

        # F-step
        F1 = f_step(F, Z, Y, FBM, nl, nc, Mask)
        F1_mat = ref['F_iter1']

        # Handle SVD sign ambiguity
        for col in range(r):
            if np.dot(F1[:, col], F1_mat[:, col]) < 0:
                F1_col = -F1[:, col]
            else:
                F1_col = F1[:, col]
            np.testing.assert_allclose(F1_col, F1_mat[:, col], atol=1e-4,
                                       err_msg=f"F_iter1 column {col} mismatch")


class TestPhase5Metrics:
    """Phase 5: Validate final metrics match MATLAB."""

    def test_final_metrics(self, ref, dataset):
        from s2sharp import s2sharp

        data = dataset
        Yim = data['Yim']
        bands = [Yim[i].astype(np.float64) for i in range(len(Yim))]
        Xm_im = data['Xm_im'].astype(np.float64)

        r = 8
        q = np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689])
        lam = 1.8998e-04

        result = s2sharp(bands, ground_truth=Xm_im, r=r, lam=lam, q=q, cd_iter=10)

        SAMm_mat = float(ref['SAMm_final'])
        aSSIM_mat = float(ref['aSSIM_final'])
        RMSE_mat = float(ref['RMSE_final'])
        ERGAS_20m_mat = float(ref['ERGAS_20m_final'])

        np.testing.assert_allclose(result.SAMm[-1], SAMm_mat, rtol=1e-3,
                                   err_msg="Final SAM mismatch")
        np.testing.assert_allclose(result.aSSIM[-1], aSSIM_mat, rtol=1e-3,
                                   err_msg="Final aSSIM mismatch")
        np.testing.assert_allclose(result.RMSE[-1], RMSE_mat, rtol=1e-3,
                                   err_msg="Final RMSE mismatch")
        np.testing.assert_allclose(result.ERGAS_20m[-1], ERGAS_20m_mat, rtol=1e-3,
                                   err_msg="Final ERGAS_20m mismatch")
