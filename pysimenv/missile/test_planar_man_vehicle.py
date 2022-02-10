import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarManVehicle2dof, PlanarNonManVehicle2dof


print("== Test for PlanarManVehicle2dof ==")
manVehicle = PlanarManVehicle2dof(
    initial_state=[-5000., 0., 200., np.deg2rad(30.)])

u = np.array([0., -5.])
dt = 0.01
time = 30.

simulator = Simulator(manVehicle)
simulator.propagate(dt, time, True, u=u)

manVehicle.default_plot()
manVehicle.plot_path()
plt.show()

print("== Test for PlanarNonManVehicle2dof ==")
nonManVehicle = PlanarNonManVehicle2dof(
    initial_state=[0., 0., 10., np.deg2rad(175.)]
)
simulator = Simulator(nonManVehicle)
simulator.propagate(dt, time, True)

nonManVehicle.default_plot()
nonManVehicle.plot_path()
plt.show()
