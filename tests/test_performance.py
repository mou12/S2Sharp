"""Performance benchmark test for S2Sharp."""

import os
import time

import numpy as np
import pytest
import scipy.io

from s2sharp import s2sharp


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


@pytest.mark.slow
def test_performance_benchmark(aviris_data):
    """Full Aviris benchmark: prints wall-clock time for regression tracking."""
    bands, Xm_im = aviris_data

    start = time.perf_counter()
    result = s2sharp(
        bands,
        ground_truth=Xm_im,
        r=8,
        lam=1.8998e-04,
        q=np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689]),
        cd_iter=2,
    )
    elapsed = time.perf_counter() - start

    print(f"\n{'='*60}")
    print(f"S2Sharp Performance Benchmark")
    print(f"{'='*60}")
    print(f"Total wall-clock time:    {elapsed:.1f}s")
    print(f"Internal Time metric:     {result.Time:.1f}s")
    print(f"CD iterations:            {len(result.SAMm)}")
    print(f"Time per CD iteration:    {elapsed / len(result.SAMm):.1f}s")
    print(f"Final SAMm:               {result.SAMm[-1]:.4f}")
    print(f"Final aSSIM:              {result.aSSIM[-1]:.4f}")
    print(f"Final RMSE:               {result.RMSE[-1]:.4f}")
    print(f"{'='*60}")

    assert result.image.ndim == 3
    assert result.image.shape[2] == 12
