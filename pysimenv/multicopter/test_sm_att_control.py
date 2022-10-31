import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.common.orientation import euler_angles_to_rotation
from pysimenv.multicopter.base import MulticopterDyn
from pysimenv.multicopter.model import QuadBase
from pysimenv.multicopter.control import SMAttControl


def main():
    model_params = {'m': 2., 'J': np.diag([1.25, 1.25, 2.50]),
                    'D_v': np.diag([1.2, 1.2, 1.6]), 'D_omega': np.diag([0.06, 0.06, 0.1])}
    R_bi = euler_angles_to_rotation(np.deg2rad([10., 10., 20.]))
    R_ib = np.transpose(R_bi)
    dyn = MulticopterDyn(p_0=np.array([1., 0., 0.]), v_0=np.zeros(3), R_0=R_ib, omega_0=np.zeros(3),
                         **model_params, name='dynamic model')
    control = SMAttControl(**model_params, c=np.array([10., 10., 10.]), k=np.array([5., 5., 5.]))
    model = QuadBase(dyn=dyn, control=control)

    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=20., save_history=True, eta_d=np.zeros(3))
    dyn.plot_euler_angles(show=True)


if __name__ == "__main__":
    main()
