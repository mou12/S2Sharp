"""Sentinel-2 band constants and default algorithm parameters."""

import numpy as np

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
NUM_BANDS = 12

# Subsampling factors (in pixels) for each band
DOWNSAMPLE_FACTORS = np.array([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])

# Modulation Transfer Function values for each band
MTF = np.array([0.32, 0.26, 0.28, 0.24, 0.38, 0.34, 0.34, 0.26, 0.33, 0.26, 0.22, 0.23])

# Default algorithm parameters
DEFAULT_RANK = 7
DEFAULT_LAMBDA = 0.005
DEFAULT_Q_R7 = np.array([1, 1.5, 4, 8, 15, 15, 20])
DEFAULT_CDITER = 10
DEFAULT_KERNEL_SIZE = (12, 12)
DEFAULT_BORDER = 2
