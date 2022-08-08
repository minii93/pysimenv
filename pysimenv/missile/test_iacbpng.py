import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarMissile2dof, PlanarNonManVehicle2dof
from pysimenv.missile.guidance import IACBPNG
from pysimenv.missile.engagement import IACBPNGEngagement


def main():
    theta_M_0 = np.deg2rad(30.)
    theta_T = 0.

    missile = PlanarMissile2dof(
        [0., 0., 200., theta_M_0]
    )
    target = PlanarNonManVehicle2dof(
        [3000., 0., 30., theta_T]
    )
    missile.name = "missile"
    target.name = "target"

    theta_M_f = np.deg2rad(-40.)
    sigma_max = np.deg2rad(45.)
    bpng = IACBPNG(N=3., tau=1., theta_M_0=theta_M_0, theta_M_f=theta_M_f, theta_T=theta_T,
                   lam_0=0., sigma_max=sigma_max, b_max=0.35)
    model = IACBPNGEngagement(missile, target, bpng)
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=30., save_history=True)

    model.plot_path()
    model.plot_rel_kin()
    model.report()
    model.report_miss_distance()
    model.report_impact_angle()
    model.report_impact_time()
    bpng.plot_bias()
    bpng.plot_bias_integral()

    plt.show()


if __name__ == "__main__":
    main()
