import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarMissile2dof, PlanarNonManVehicle2dof
from pysimenv.missile.engagement import PurePNG2dimEngagement


def main():
    print("== Two dimensional pure PNG engagement for a stationary target ==")
    missile = PlanarMissile2dof(
        initial_state=[-5000., 3000., 300., np.deg2rad(-15.)]
    )
    target = PlanarNonManVehicle2dof(
        initial_state=[0., 0., 20., 0.]
    )
    missile.name = "missile"
    target.name = "target"

    # Method 1
    model = PurePNG2dimEngagement(missile, target)
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=30., save_history=True)

    model.plot_path()
    model.plot_rel_kin()
    model.report()
    model.save_log_file('./data/planar_png/')
    # model.load_log_file('./data/planar_png/')

    plt.show()


if __name__ == "__main__":
    main()
