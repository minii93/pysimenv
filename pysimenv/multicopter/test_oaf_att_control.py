import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.common.orientation import euler_angles_to_rotation
from pysimenv.multicopter.base import MulticopterDyn
from pysimenv.multicopter.model import QuadXEffector, QuadBase
from pysimenv.multicopter.control import OAFAttControl


def main():
    model_params = {'m': 1.023, 'J': np.diag([9.5, 9.5, 1.86])*0.001,
                    'D_v': np.diag([1.2, 1.2, 1.6]), 'D_omega': np.diag([0.06, 0.06, 0.1])}

    eta_0 = np.deg2rad([15., -10., 0.])
    R_bi_0 = euler_angles_to_rotation(eta_0)
    R_ib_0 = R_bi_0.transpose()
    dyn = MulticopterDyn(p_0=np.array([0., 0., -5.]), v_0=np.zeros(3), R_0=R_ib_0, omega_0=np.zeros(3), **model_params,
                         name='dynamic model')
    effector = QuadXEffector(d_phi=0.15, d_theta=0.15, c_tau_f=0.0196)
    control = OAFAttControl(**model_params, G=effector.B,
                               K_p=np.array([121., 121., 2.25]), K_d=np.array([22., 22., 3.]))

    model = QuadBase(dyn=dyn, control=control, effector=effector)
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=10., save_history=True, fault_ind=2, zeta_d=np.array([0., 0., -2.]))
    dyn.plot_path()
    dyn.plot_euler_angles()
    dyn.default_plot(show=True)


if __name__ == "__main__":
    main()

