"""
CADLib - A library for CAD representation design automation
"""

from .curves import Line, Arc, Circle, CurveBase
from .sketch import Loop, Profile, SketchBase
from .extrude import Extrude, CADSequence, CoordSystem
from .visualize import (
    vec2CADsolid, 
    create_CAD, 
    create_by_extrude, 
    CADsolid2pc
)
from .macro import *
from .math_utils import *

__version__ = "1.0.0"
__all__ = [
    "Line", "Arc", "Circle", "CurveBase",
    "Loop", "Profile", "SketchBase", 
    "Extrude", "CADSequence", "CoordSystem",
    "vec2CADsolid", "create_CAD", "create_by_extrude", "CADsolid2pc"
]