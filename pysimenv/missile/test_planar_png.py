import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarMissile, PlanarMovingTarget
from pysimenv.missile.guidance import PurePNG2dim
from pysimenv.missile.engagement import Engagement2dim


def main():
    missile = PlanarMissile(p_0=[-5000., 3000.], V_0=300., gamma_0=np.deg2rad(-15.), name="missile")
    target = PlanarMovingTarget(p_0=[0., 0.], V_0=20., gamma_0=0., name="target")

    guidance = PurePNG2dim(N=3.0)
    model = Engagement2dim(missile, target, guidance)
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=30., save_history=True)

    simulator.save_sim_log('./data/planar_png/')
    model.report()
    model.plot_path()
    model.plot_rel_kin(show=True)


if __name__ == "__main__":
    main()
