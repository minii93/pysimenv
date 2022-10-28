import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.multicopter.model import MulticopterDyn, QuadBase
from pysimenv.multicopter.control import BSControl


def main():
    model_params = {'m': 1.023, 'J': np.diag([9.5, 9.5, 1.86])*1e-3}
    alpha = np.array([16., 14., 16., 14., 16., 14., 2.5, 0.5, 2.5, 0.5, 2.5, 0.5])

    dyn = MulticopterDyn(p_0=np.zeros(3), v_0=np.zeros(3), R_0=np.identity(3), omega_0=np.zeros(3), **model_params,
                         name='dynamic model')
    control = BSControl(alpha=alpha, **model_params, name='controller')
    model = QuadBase(dyn=dyn, control=control, name='quad-rotor')

    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=10., save_history=True, sigma_d=np.array([2., 2., -2., np.deg2rad(15.)]))
    model.dyn.plot_path()
    model.dyn.plot_euler_angles()
    model.dyn.default_plot(show=True)


if __name__ == "__main__":
    main()
