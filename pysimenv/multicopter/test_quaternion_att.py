import numpy as np
from pysimenv.core.base import SimObject
from pysimenv.multicopter.model import MulticopterDyn
from pysimenv.multicopter.control import QuaternionAttControl
from pysimenv.common.orientation import euler_angles_to_rotation
from pysimenv.common.model import FlatEarthEnv
from pysimenv.core.simulator import Simulator


class Model(SimObject):
    def __init__(self):
        super(Model, self).__init__()
        model_params = {'m': 1.3, 'J': np.diag([0.0119, 0.0119, 0.0219])}

        R_vi = euler_angles_to_rotation(np.deg2rad([10., 10., 20.]))
        R_iv = np.transpose(R_vi)
        self.dyn = MulticopterDyn(p_0=np.array([1., 2., 0.]), v_0=np.zeros(3), R_0=R_iv, omega_0=np.array([0., 0., 1.]),
                                  **model_params, name='dynamic model')

        Q = np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2
        R = 1e-4*np.identity(3)
        K = QuaternionAttControl.gain(Q, R)
        self.att_control = QuaternionAttControl(J=model_params['J'], K=K, name='controller')

        self._add_sim_objs([self.dyn, self.att_control])

    def _forward(self, q_d: np.ndarray, omega_d: np.ndarray = np.zeros(3)):
        tau = self.att_control.forward(self.dyn, q_d, omega_d)
        f = self.dyn.m*FlatEarthEnv.grav_accel

        self.dyn.forward(f=f, tau=tau)


def main():
    q_d = np.array([1., 0., 0., 0.])
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True, q_d=q_d)
    model.dyn.default_plot(show=True)


if __name__ == "__main__":
    main()
