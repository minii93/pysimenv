import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.missile.model import PlanarMissile


def main():
    missile = PlanarMissile(p_0=[-5000., 0.], V_0=200., gamma_0=np.deg2rad(30.))
    simulator = Simulator(missile)
    simulator.propagate(dt=0.01, time=50., save_history=True, a_M_cmd=-5.)
    missile.plot_kin()
    missile.plot_path(show=True, label="missile")


if __name__ == "__main__":
    main()

