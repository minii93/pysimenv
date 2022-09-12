import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PitchDyn, PlanarMissileWithPitch, PlanarMovingTarget
from pysimenv.missile.control import PitchAP
from pysimenv.missile.guidance import PurePNG2dim
from pysimenv.missile.engagement import Engagement2dim


def main():
    aero_params = dict(L_alp=1270., L_delta=80., M_alp=-74., M_q=-5., M_delta=160.)

    pitch_dyn = PitchDyn(x_0=np.zeros(3), V=300., **aero_params)
    pitch_ap = PitchAP(V=300., **aero_params, tau=0.02)
    missile = PlanarMissileWithPitch(p_0=[-5000., 0.], V_0=300., gamma_0=np.deg2rad(30.),
                                     pitch_dyn=pitch_dyn, pitch_ap=pitch_ap, tau=0.02, name="missile")
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
