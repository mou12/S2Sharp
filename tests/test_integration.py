"""End-to-end integration test using the Aviris dataset."""

import os
import numpy as np
import pytest
import scipy.io

from s2sharp import s2sharp


# Path to test data
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Aviris_cell_3.mat')


@pytest.fixture
def aviris_data():
    """Load the Aviris test dataset."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("Test data not found: Data/Aviris_cell_3.mat")

    data = scipy.io.loadmat(DATA_PATH, squeeze_me=True, simplify_cells=True)

    Yim = data['Yim']
    bands = [Yim[i].astype(np.float64) for i in range(len(Yim))]
    Xm_im = data['Xm_im'].astype(np.float64)

    return bands, Xm_im


class TestIntegration:
    """End-to-end tests.

    Note: These tests use the real 408x408 dataset and involve heavy FFT
    computations. Each Z-step takes ~40s, F-step ~80s on typical hardware.
    """

    @pytest.mark.slow
    def test_s2sharp_runs(self, aviris_data):
        """S2sharp should run without errors on the test dataset (2 CD iters)."""
        bands, Xm_im = aviris_data

        result = s2sharp(
            bands,
            ground_truth=Xm_im,
            r=8,
            lam=1.8998e-04,
            q=np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689]),
            cd_iter=2,
        )

        assert result.image.ndim == 3
        assert result.image.shape[2] == 12
        assert len(result.SAMm) == 2
        assert len(result.SRE) == 2
        assert result.Time > 0

        # Metrics should improve across iterations
        assert result.SAMm[-1] <= result.SAMm[0]
        assert result.aSSIM[-1] >= result.aSSIM[0]

    def test_g_step_only(self, aviris_data):
        """g_step_only mode: single Z-step without F-step optimization."""
        bands, Xm_im = aviris_data

        result = s2sharp(
            bands,
            ground_truth=Xm_im,
            r=8,
            lam=1.8998e-04,
            q=np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689]),
            g_step_only=True,
        )

        assert result.image.ndim == 3
        assert result.image.shape[2] == 12
        assert len(result.SAMm) == 1

        # Quality should be reasonable even with single iteration
        assert result.SAMm[0] < 10.0
        assert result.aSSIM[0] > 0.9
        assert result.RMSE[0] < 5.0

    def test_no_ground_truth(self, aviris_data):
        """S2sharp should work without ground truth (no metrics)."""
        bands, _ = aviris_data

        result = s2sharp(
            bands,
            r=8,
            lam=1.8998e-04,
            q=np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689]),
            g_step_only=True,
        )

        assert result.image.ndim == 3
        assert result.image.shape[2] == 12
        assert len(result.SAMm) == 0  # No metrics without ground truth
        assert result.image.min() >= 0  # Image values should be non-negative
