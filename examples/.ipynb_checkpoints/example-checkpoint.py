"""Example demonstrating S2Sharp on the Aviris simulated Sentinel-2 dataset.

Equivalent to the MATLAB example.m script.
"""

import os
import numpy as np
import scipy.io

from s2sharp import s2sharp


def main():
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Aviris_cell_3.mat')
    data = scipy.io.loadmat(data_path, squeeze_me=True, simplify_cells=True)

    # Extract bands (cell array -> list of 2D arrays)
    Yim = data['Yim']
    if isinstance(Yim, np.ndarray) and Yim.dtype == object:
        bands = [Yim[i].astype(np.float64) for i in range(len(Yim))]
    else:
        bands = [Yim[i].astype(np.float64) for i in range(len(Yim))]

    # Ground truth
    Xm_im = data['Xm_im'].astype(np.float64)

    # Parameters
    r = 8
    q = np.array([1, 0.3851, 6.9039, 19.9581, 47.8967, 27.5518, 2.7100, 34.8689])
    ni = 10
    lam = 1.8998e-04

    # Run S2Sharp
    result = s2sharp(
        bands,
        ground_truth=Xm_im,
        r=r,
        lam=lam,
        q=q,
        cd_iter=ni,
    )

    # Output results
    d = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    # Indices for non-10m bands (0-based): B1(0), B5(4), B6(5), B7(6), B8A(8), B9(9), B11(10), B12(11)
    sre_bands = [0, 4, 5, 6, 8, 9, 10, 11]

    S2sharp_SRE = result.SRE[-1][sre_bands]
    S2sharp_SAM = result.SAMm[-1]
    S2sharp_aSRE = np.mean(S2sharp_SRE)
    S2sharp_RMSE = result.RMSE[-1]
    S2sharp_aSSIM = result.aSSIM[-1]
    S2sharp_ERGAS_60m = result.ERGAS_60m[-1]
    S2sharp_ERGAS_20m = result.ERGAS_20m[-1]
    S2sharp_time = result.Time

    print(f"S2sharp: Best lambda={lam}")
    print(f"S2sharp: SAM={S2sharp_SAM:.4f}")
    print(f"Average SRE = {S2sharp_aSRE:.4f}")
    print(f"S2sharp aSSIM = {S2sharp_aSSIM:.4f}")
    print(f"S2sharp RMSE = {S2sharp_RMSE:.6f}")
    print(f"S2sharp time = {S2sharp_time:.2f}")
    print("S2sharp: SRE:")
    print("B1    B5    B6    B7    B8a   B9    B11   B12")
    print(" ".join(f"{v:.2f}" for v in S2sharp_SRE))


if __name__ == "__main__":
    main()
