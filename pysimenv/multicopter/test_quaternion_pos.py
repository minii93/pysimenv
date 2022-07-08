import numpy as np
from pysimenv.core.system import MultipleSystem
from pysimenv.multicopter.model import QuadrotorDynModel
from pysimenv.multicopter.control import QuaternionPosControl, QuaternionAttControl
from pysimenv.common.orientation import rotation_to_quaternion
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
        self.quadrotor = QuadrotorDynModel([pos, vel, R_iv, omega], m, J)

        Q_att = np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2
        R_att = 1e-4*np.identity(3)
        K_att = QuaternionAttControl.gain(Q_att, R_att)
        self.att_control = QuaternionAttControl(J, K_att)

        Q_pos = np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2
        R_pos = np.identity(3)
        K_pos = QuaternionPosControl.gain(Q_pos, R_pos)
        self.pos_control = QuaternionPosControl(m, K_pos)

        self.attach_sim_objects([self.quadrotor, self.att_control, self.pos_control])

    def forward(self, p_d: np.ndarray, v_d: np.ndarray = np.zeros(3)):
        quad_states = self.quadrotor.state

        p = quad_states[0]
        v = quad_states[1]
        R_iv = quad_states[2]
        omega = quad_states[3]

        q = rotation_to_quaternion(np.transpose(R_iv))

        f, q_d, omega_d = self.pos_control.forward(p, v, p_d, v_d)
        tau = self.att_control.forward(q, omega, q_d, omega_d)

        self.quadrotor.forward(np.array([f, tau[0], tau[1], tau[2]]))


def main():
    print("== Test for position control based on quaternion ==")
    p_d = np.array([0., 0., 1.])
    v_d = np.array([0., 0., 0.])
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True, p_d, v_d)
    model.quadrotor.default_plot(show=True)


if __name__ == "__main__":
    main()
