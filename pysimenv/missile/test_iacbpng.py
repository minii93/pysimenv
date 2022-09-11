import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarMissile, PlanarMovingTarget
from pysimenv.missile.guidance import IACBPNG
from pysimenv.missile.engagement import Engagement2dim


def main():
    theta_M_0 = np.deg2rad(30.)
    theta_T = 0.

    missile = PlanarMissile(p_0=[0., 0.], V_0=200., gamma_0=theta_M_0, name="missile")
    target = PlanarMovingTarget(p_0=[3000., 0.], V_0=30., gamma_0=theta_T, name="target")

    theta_M_f = np.deg2rad(-40.)
    sigma_max = np.deg2rad(45.)
    bpng = IACBPNG(N=3., tau=1., theta_M_0=theta_M_0, theta_M_f=theta_M_f, theta_T=theta_T,
                   lam_0=0., sigma_max=sigma_max, b_max=0.35)
    model = Engagement2dim(missile, target, bpng)
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=30., save_history=True)

    model.report()
    bpng.plot_bias()
    bpng.plot_bias_integral()
    model.plot_path()
    model.plot_rel_kin(show=True)


if __name__ == "__main__":
    main()
