import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.multicopter.base import MulticopterDyn
from pysimenv.multicopter.model import QuadXEffector, QuadBase
from pysimenv.multicopter.control import OAFControl


def main():
    model_params = {'m': 1.023, 'J': np.diag([9.5, 9.5, 1.86])*0.001,
                    'D_v': np.diag([1.2, 1.2, 1.6]), 'D_omega': np.diag([0.06, 0.06, 0.1])}

    dyn = MulticopterDyn(p_0=np.array([0., 0., -5.]), v_0=np.zeros(3), R_0=np.identity(3), omega_0=np.zeros(3), **model_params,
                         name='dynamic model')
    effector = QuadXEffector(d_phi=0.15, d_theta=0.15, c_tau_f=0.0196)
    control = OAFControl(**model_params, G=effector.B,
                            K_p=np.array([1.24, 1.24, 1.24, 225., 225.]),
                            K_d=np.array([1., 1., 1., 30., 30.]))

    model = QuadBase(dyn=dyn, control=control, effector=effector)
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=20., save_history=True, fault_ind=2, p_d=np.array([2., 2., 0.]))
    dyn.plot_path()
    dyn.plot_euler_angles()
    control.plot_desired_att()
    dyn.default_plot(show=True)


if __name__ == "__main__":
    main()
