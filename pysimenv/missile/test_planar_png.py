import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarMissile2dof, PlanarNonManVehicle2dof
from pysimenv.missile.engagement import PurePNG2dimEngagement


print("== Two dimensional pure PNG engagement for a stationary target ==")

dt = 0.01
final_time = 30.
missile = PlanarMissile2dof(
    [-5000., 3000., 300., np.deg2rad(-15.)]
)
target = PlanarNonManVehicle2dof(
    [0., 0., 20., 0.]
)
target.name = "target"

# Method 1
model = PurePNG2dimEngagement(missile, target)
simulator = Simulator(model)
simulator.propagate(dt, final_time, True)

model.plot()
model.report()

plt.show()
