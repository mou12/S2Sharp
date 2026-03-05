"""Benchmark warm-start vs cold-start CG across CD iterations."""

import os
import time

import numpy as np
import pytest
import scipy.io

from s2sharp import s2sharp

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Aviris_cell_3.mat')


@pytest.fixture(scope="module")
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
class TestWarmStart:
    """Compare warm-start vs cold-start CG performance."""

    def test_warm_start_reduces_cg_iterations(self, aviris_data):
        """Verify warm-start CG uses fewer iterations than cold-start."""
        bands, Xm_im = aviris_data

        kwargs = dict(
            r=8,
            lam=1.8998e-04,
            q=np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689]),
            cd_iter=10,
            ground_truth=Xm_im,
        )

        result_warm = s2sharp(bands, warm_start=True, **kwargs)
        result_cold = s2sharp(bands, warm_start=False, **kwargs)

        warm_iters = result_warm.cg_iterations
        cold_iters = result_cold.cg_iterations

        print("\n=== CG Iteration Comparison ===")
        print(f"{'CD iter':<10} {'Warm':<10} {'Cold':<10} {'Ratio':<10}")
        print("-" * 40)
        for i, (w, c) in enumerate(zip(warm_iters, cold_iters), 1):
            ratio = w / c if c > 0 else float('nan')
            print(f"{i:<10} {w:<10} {c:<10} {ratio:<10.3f}")

        total_warm = sum(warm_iters)
        total_cold = sum(cold_iters)
        print(f"\n{'Total':<10} {total_warm:<10} {total_cold:<10} {total_warm/total_cold:<10.3f}")

        # First iteration should be identical (both start from zeros)
        assert warm_iters[0] == cold_iters[0], "First iteration should be identical"

        # Warm-start should use fewer total CG iterations (after iter 1)
        assert sum(warm_iters[1:]) <= sum(cold_iters[1:]), (
            f"Warm-start should use fewer CG iterations after iter 1: "
            f"warm={sum(warm_iters[1:])}, cold={sum(cold_iters[1:])}"
        )

        # Both should produce similar final quality (within tolerance).
        # Warm-start may actually produce slightly better results since CG
        # starts closer to the optimum, allowing more precise convergence.
        np.testing.assert_allclose(
            result_warm.SAMm[-1], result_cold.SAMm[-1], rtol=5e-2,
            err_msg="Final SAM should be similar between warm and cold start"
        )
        np.testing.assert_allclose(
            result_warm.aSSIM[-1], result_cold.aSSIM[-1], rtol=5e-2,
            err_msg="Final aSSIM should be similar between warm and cold start"
        )

    def test_warm_start_timing(self, aviris_data):
        """Measure wall-clock speedup from warm-start."""
        bands, Xm_im = aviris_data

        kwargs = dict(
            r=8,
            lam=1.8998e-04,
            q=np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689]),
            cd_iter=10,
            ground_truth=Xm_im,
        )

        t0 = time.perf_counter()
        result_warm = s2sharp(bands, warm_start=True, **kwargs)
        t_warm = time.perf_counter() - t0

        t0 = time.perf_counter()
        result_cold = s2sharp(bands, warm_start=False, **kwargs)
        t_cold = time.perf_counter() - t0

        print(f"\n=== Timing Comparison ===")
        print(f"Warm-start: {t_warm:.1f}s")
        print(f"Cold-start: {t_cold:.1f}s")
        print(f"Speedup:    {t_cold/t_warm:.2f}x")
        print(f"\nTotal CG iters (warm): {sum(result_warm.cg_iterations)}")
        print(f"Total CG iters (cold): {sum(result_cold.cg_iterations)}")

        print(f"\n=== Final Metrics ===")
        print(f"SAM  (warm): {result_warm.SAMm[-1]:.4f}  (cold): {result_cold.SAMm[-1]:.4f}")
        print(f"aSSIM(warm): {result_warm.aSSIM[-1]:.4f}  (cold): {result_cold.aSSIM[-1]:.4f}")
        print(f"RMSE (warm): {result_warm.RMSE[-1]:.4f}  (cold): {result_cold.RMSE[-1]:.4f}")
