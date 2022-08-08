import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarMissile2dof


def main():
    print("== Test for PlanarMissile2dof ==")
    missile = PlanarMissile2dof(
        initial_state=[-5000., 0., 200., np.deg2rad(30.)])
    simulator = Simulator(missile)
    simulator.propagate(dt=0.01, time=50., save_history=True, a_M_cmd=np.array([0., -5.]))

    missile.report()
    missile.plot()
    missile.plot_path(show=True)


if __name__ == "__main__":
    main()

