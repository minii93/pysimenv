import numpy as np
from pysimenv.core.base import SimObject
from pysimenv.multicopter.base import MulticopterDyn
from pysimenv.multicopter.control import QuaternionPosControl, QuaternionAttControl
from pysimenv.core.simulator import Simulator


class Model(SimObject):
    def __init__(self):
        super(Model, self).__init__()
        model_params = {'m': 1.3, 'J': np.diag([0.0119, 0.0119, 0.0219])}

        self.dyn = MulticopterDyn(p_0=np.array([1., 2., 0.]), v_0=np.zeros(3), R_0=np.identity(3), omega_0=np.zeros(3),
                                  **model_params, name='dynamic_model')

        K_att = QuaternionAttControl.gain(
            Q=np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2,
            R=1e-4*np.identity(3)
        )
        self.att_control = QuaternionAttControl(J=model_params['J'], K=K_att)

        K_pos = QuaternionPosControl.gain(
            Q=np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2,
            R=np.identity(3)
        )
        self.pos_control = QuaternionPosControl(m=model_params['m'], K=K_pos)

        self._add_sim_objs([self.dyn, self.att_control, self.pos_control])

    def _forward(self, p_d: np.ndarray, v_d: np.ndarray = np.zeros(3)):
        f, q_d, omega_d = self.pos_control.forward(self.dyn, p_d, v_d)
        tau = self.att_control.forward(self.dyn, q_d, omega_d)

        self.dyn.forward(f=f, tau=tau)


def main():
    p_d = np.array([0., 0., 1.])
    v_d = np.array([0., 0., 0.])
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True, p_d=p_d, v_d=v_d)
    model.dyn.default_plot(show=True)


if __name__ == "__main__":
    main()
