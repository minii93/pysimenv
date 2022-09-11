import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PitchDynLatAccel, PlanarMissileWithPitch, PlanarMovingTarget
from pysimenv.missile.guidance import PurePNG2dim
from pysimenv.missile.engagement import Engagement2dim


def main():
    pitch_dyn = PitchDynLatAccel(x_0=np.zeros(3), L_alp=1270., L_delta=80., M_alp=-74., M_q=-5., M_delta=160.)
    missile = PlanarMissileWithPitch(p_0=[-5000., 3000.], V_0=300., gamma_0=np.deg2rad(-15.),
                                     pitch_dyn=pitch_dyn, name="missile")
    target = PlanarMovingTarget(p_0=[0., 0.], V_0=20., gamma_0=0., name="target")

    guidance = PurePNG2dim(N=3.0)
    model = Engagement2dim(missile, target, guidance)
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=30., save_history=True)

    model.report()
    model.plot_path()
    pitch_dyn.plot(show=True)


if __name__ == "__main__":
    main()
