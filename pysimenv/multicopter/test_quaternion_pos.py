import numpy as np
from pysimenv.core.system import MultipleSystem
from pysimenv.multicopter.model import MulticopterDynamic
from pysimenv.multicopter.control import QuaternionPosControl, QuaternionAttControl
from pysimenv.core.simulator import Simulator


class Model(MultipleSystem):
    def __init__(self):
        super(Model, self).__init__()

        m = 1.3
        J = np.diag([0.0119, 0.0119, 0.0219])

        pos = np.array([1., 2., 0.])
        vel = np.zeros(3)
        R_iv = np.identity(3)
        omega = np.zeros(3)
        self.quadrotor = MulticopterDynamic([pos, vel, R_iv, omega], m, J)

        K_att = QuaternionAttControl.gain(
            Q=np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2,
            R=1e-4*np.identity(3)
        )
        self.att_control = QuaternionAttControl(J, K_att)

        K_pos = QuaternionPosControl.gain(
            Q=np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2,
            R=np.identity(3)
        )
        self.pos_control = QuaternionPosControl(m, K_pos)

        self.attach_sim_objects([self.quadrotor, self.att_control, self.pos_control])

    def forward(self, p_d: np.ndarray, v_d: np.ndarray = np.zeros(3)):
        p = self.quadrotor.pos
        v = self.quadrotor.vel
        q = self.quadrotor.quaternion
        omega = self.quadrotor.ang_vel

        f, q_d, omega_d = self.pos_control.forward(p, v, p_d, v_d)
        tau = self.att_control.forward(q, omega, q_d, omega_d)

        self.quadrotor.forward(u=np.array([f, tau[0], tau[1], tau[2]]))


def main():
    print("== Test for position control based on quaternion ==")
    p_d = np.array([0., 0., 1.])
    v_d = np.array([0., 0., 0.])
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True, p_d=p_d, v_d=v_d)
    model.quadrotor.default_plot(show=True)


if __name__ == "__main__":
    main()
