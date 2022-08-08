import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarMissile2dof, PlanarNonManVehicle2dof
from pysimenv.missile.guidance import PurePNG2dim
from pysimenv.missile.engagement import Engagement2dim


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
    guidance = PurePNG2dim(N=3.0)
    model = Engagement2dim(missile, target, guidance)
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=30., save_history=True)

    model.plot_path()
    model.plot_rel_kin()
    model.report()
    model.report_miss_distance()
    model.report_impact_angle()
    model.report_impact_time()
    model.save_log_file('./data/planar_png/')
    # model.load_log_file('./data/planar_png/')

    plt.show()


if __name__ == "__main__":
    main()
