"""maya-encoding: Maya-inspired numerical encodings for machine learning.

Two encoding systems:
- VFDEncoder: Vigesimal Feature Decomposition for tabular numeric features
- MayaCalendarEncoder: Maya calendar-based cyclical encoding for temporal features
"""

from maya_encoding._version import __version__
from maya_encoding.core.vigesimal import maya_decompose, to_bars_dots, to_vigesimal
from maya_encoding.mce.encoder import MayaCalendarEncoder
from maya_encoding.vfd.encoder import VFDEncoder

__all__ = [
    "__version__",
    "VFDEncoder",
    "MayaCalendarEncoder",
    "maya_decompose",
    "to_vigesimal",
    "to_bars_dots",
]
