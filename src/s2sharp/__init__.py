"""S2Sharp: Sentinel-2 sharpening using a reduced-rank method.

Reference:
    Sentinel-2 Sharpening Using a Reduced-Rank Method,
    M.O. Ulfarsson et al., IEEE Trans. Geoscience and Remote Sensing, 2019.
"""

from .core import S2SharpResult, s2sharp

__all__ = ["s2sharp", "S2SharpResult"]
__version__ = "1.0.0"
