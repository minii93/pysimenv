import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarManVehicle2dof, PlanarNonManVehicle2dof


def main():
    print("== Test for PlanarManVehicle2dof ==")
    man_vehicle = PlanarManVehicle2dof(
        initial_state=[-5000., 0., 200., np.deg2rad(30.)])
    simulator = Simulator(man_vehicle)
    simulator.propagate(dt=0.01, time=30., save_history=True, u=np.array([0., -5.]))
    man_vehicle.default_plot()
    man_vehicle.plot_path(show=True)

    print("== Test for PlanarNonManVehicle2dof ==")
    non_man_vehicle = PlanarNonManVehicle2dof(
        initial_state=[0., 0., 10., np.deg2rad(175.)]
    )
    simulator = Simulator(non_man_vehicle)
    simulator.propagate(dt=0.01, time=30., save_history=True)
    non_man_vehicle.default_plot()
    non_man_vehicle.plot_path(show=True)


if __name__ == "__main__":
    main()
