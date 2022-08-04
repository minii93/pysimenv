"""
Description for Package
"""
from pysimenv.missile.model import PlanarManVehicle2dof, PlanarNonManVehicle2dof, PlanarMissile2dof
from pysimenv.missile.engagement import Engagement2dim, PurePNG2dimEngagement, IACBPNGEngagement
from pysimenv.missile.guidance import PurePNG2dim, IACBPNG
from pysimenv.missile.util import RelKin2dim, CloseDistCond, miss_distance

__all__ = ['missile']