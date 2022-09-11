"""
Description for Package
"""
from pysimenv.missile.model import PlanarKin, PitchDyn, PitchDynLatAccel, PlanarVehicle, PlanarMissile,\
    PlanarMissileWithPitch, PlanarMovingTarget
from pysimenv.missile.engagement import Engagement2dim
from pysimenv.missile.guidance import PurePNG2dim, IACBPNG
from pysimenv.missile.util import RelKin2dim, CloseDistCond, miss_distance

__all__ = ['missile']