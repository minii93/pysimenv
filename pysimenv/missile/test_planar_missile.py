import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarMissile2dof


print("== Test for PlanarMissile2dof ==")
initial_state = [-5000., 0., 200., np.deg2rad(30.)]
missile = PlanarMissile2dof(initial_state)

a_M = np.array([0., -5.])
dt = 0.01
time = 50.

simulator = Simulator(missile)
simulator.propagate(dt, time, True, a_M=a_M)

missile.report()
missile.plot()
missile.plot_path()
plt.show()
