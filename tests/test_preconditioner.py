"""Benchmark tests comparing CG vs PCG (preconditioned CG) on AVIRIS data."""

import os
import time

import numpy as np
import pytest
import scipy.io

from s2sharp import s2sharp

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Aviris_cell_3.mat')

AVIRIS_KWARGS = dict(
    r=8,
    lam=1.8998e-04,
    q=np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689]),
    cd_iter=10,
)


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
class TestPreconditionerBenchmark:
    """Compare CG vs PCG performance on real data."""

    def test_pcg_reduces_iterations(self, aviris_data):
        """PCG should use fewer total CG iterations than standard CG."""
        bands, Xm_im = aviris_data

        result_cg = s2sharp(bands, warm_start=True, precondition=False,
                            ground_truth=Xm_im, **AVIRIS_KWARGS)
        result_pcg = s2sharp(bands, warm_start=True, precondition=True,
                             ground_truth=Xm_im, **AVIRIS_KWARGS)

        cg_iters = result_cg.cg_iterations
        pcg_iters = result_pcg.cg_iterations

        print("\n=== CG vs PCG Iteration Comparison (warm-start) ===")
        print(f"{'CD iter':<10} {'CG':<10} {'PCG':<10} {'Ratio':<10}")
        print("-" * 40)
        for i, (c, p) in enumerate(zip(cg_iters, pcg_iters), 1):
            ratio = p / c if c > 0 else float('nan')
            print(f"{i:<10} {c:<10} {p:<10} {ratio:<10.3f}")

        total_cg = sum(cg_iters)
        total_pcg = sum(pcg_iters)
        print(f"\n{'Total':<10} {total_cg:<10} {total_pcg:<10} {total_pcg/total_cg:<10.3f}")

        print(f"\n=== Quality Comparison ===")
        print(f"SAM   (CG): {result_cg.SAMm[-1]:.4f}  (PCG): {result_pcg.SAMm[-1]:.4f}")
        print(f"aSSIM (CG): {result_cg.aSSIM[-1]:.4f}  (PCG): {result_pcg.aSSIM[-1]:.4f}")
        print(f"RMSE  (CG): {result_cg.RMSE[-1]:.4f}  (PCG): {result_pcg.RMSE[-1]:.4f}")

        assert total_pcg <= total_cg, (
            f"PCG should use fewer total CG iterations: PCG={total_pcg}, CG={total_cg}"
        )

        # Quality should be similar
        np.testing.assert_allclose(
            result_pcg.SAMm[-1], result_cg.SAMm[-1], rtol=5e-2,
            err_msg="PCG should produce similar SAM to CG"
        )

    def test_pcg_reduces_wallclock(self, aviris_data):
        """PCG should not be slower than CG (allow 10% margin)."""
        bands, Xm_im = aviris_data

        t0 = time.perf_counter()
        result_cg = s2sharp(bands, warm_start=True, precondition=False,
                            ground_truth=Xm_im, **AVIRIS_KWARGS)
        t_cg = time.perf_counter() - t0

        t0 = time.perf_counter()
        result_pcg = s2sharp(bands, warm_start=True, precondition=True,
                             ground_truth=Xm_im, **AVIRIS_KWARGS)
        t_pcg = time.perf_counter() - t0

        print(f"\n=== Wall-Clock Comparison ===")
        print(f"CG:      {t_cg:.1f}s  ({sum(result_cg.cg_iterations)} iters)")
        print(f"PCG:     {t_pcg:.1f}s  ({sum(result_pcg.cg_iterations)} iters)")
        print(f"Speedup: {t_cg/t_pcg:.2f}x")

        assert t_pcg <= t_cg * 1.1, (
            f"PCG should not be slower than CG: PCG={t_pcg:.1f}s, CG={t_cg:.1f}s"
        )

    def test_cold_start_pcg(self, aviris_data):
        """PCG should show biggest improvement on cold-start (first iteration)."""
        bands, Xm_im = aviris_data

        result_cg = s2sharp(bands, warm_start=False, precondition=False,
                            ground_truth=Xm_im, **AVIRIS_KWARGS)
        result_pcg = s2sharp(bands, warm_start=False, precondition=True,
                             ground_truth=Xm_im, **AVIRIS_KWARGS)

        cg_iters = result_cg.cg_iterations
        pcg_iters = result_pcg.cg_iterations

        print("\n=== Cold-Start CG vs PCG ===")
        print(f"{'CD iter':<10} {'CG':<10} {'PCG':<10} {'Ratio':<10}")
        print("-" * 40)
        for i, (c, p) in enumerate(zip(cg_iters, pcg_iters), 1):
            ratio = p / c if c > 0 else float('nan')
            print(f"{i:<10} {c:<10} {p:<10} {ratio:<10.3f}")

        total_cg = sum(cg_iters)
        total_pcg = sum(pcg_iters)
        print(f"\n{'Total':<10} {total_cg:<10} {total_pcg:<10} {total_pcg/total_cg:<10.3f}")

        assert total_pcg <= total_cg, (
            f"Cold-start PCG should use fewer iterations: PCG={total_pcg}, CG={total_cg}"
        )
