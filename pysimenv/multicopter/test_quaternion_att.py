import numpy as np
from pysimenv.multicopter.base import MulticopterDyn
from pysimenv.multicopter.model import QuadBase
from pysimenv.multicopter.control import QuaternionAttControl
from pysimenv.common.orientation import euler_angles_to_rotation
from pysimenv.core.simulator import Simulator


def main():
    model_params = {'m': 1.3, 'J': np.diag([0.0119, 0.0119, 0.0219])}

    R_vi = euler_angles_to_rotation(np.deg2rad([10., 10., 20.]))
    R_iv = np.transpose(R_vi)
    dyn = MulticopterDyn(p_0=np.array([1., 2., 0.]), v_0=np.zeros(3), R_0=R_iv, omega_0=np.array([0., 0., 1.]),
                              **model_params, name='dynamic model')

    Q = np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2
    R = 1e-4*np.identity(3)
    K = QuaternionAttControl.gain(Q, R)
    att_control = QuaternionAttControl(**model_params, K=K, name='controller')
    model = QuadBase(dyn=dyn, control=att_control)

    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True, q_d=np.array([1., 0., 0., 0.]))
    model.dyn.default_plot(show=True)


if __name__ == "__main__":
    main()
