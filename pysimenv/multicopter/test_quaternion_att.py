import numpy as np
from pysimenv.core.system import MultipleSystem
from pysimenv.multicopter.model import MulticopterDynamic
from pysimenv.multicopter.control import QuaternionAttControl
from pysimenv.common.orientation import euler_angles_to_rotation
from pysimenv.common.model import FlatEarthEnv
from pysimenv.core.simulator import Simulator


class Model(MultipleSystem):
    def __init__(self):
        super(Model, self).__init__()

        m = 1.3
        J = np.diag([0.0119, 0.0119, 0.0219])

        pos = np.array([1., 2., 0.])
        vel = np.zeros(3)
        R_vi = euler_angles_to_rotation(np.deg2rad([10., 10., 20.]))
        R_iv = np.transpose(R_vi)
        omega = np.array([0., 0., 1.])
        self.quadrotor = MulticopterDynamic([pos, vel, R_iv, omega], m, J)

        Q = np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2
        R = 1e-4*np.identity(3)
        K = QuaternionAttControl.gain(Q, R)
        self.att_control = QuaternionAttControl(J, K)

        self.attach_sim_objects([self.quadrotor, self.att_control])

    def forward(self, q_d: np.ndarray, omega_d: np.ndarray = np.zeros(3)):
        q = self.quadrotor.quaternion
        omega = self.quadrotor.ang_vel

        f = self.quadrotor.m*FlatEarthEnv.grav_accel
        tau = self.att_control.forward(q, omega, q_d, omega_d)
        self.quadrotor.forward(u=np.array([f, tau[0], tau[1], tau[2]]))


def main():
    print("== Test for attitude control based on quaternion ==")
    q_d = np.array([1., 0., 0., 0.])
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True, q_d=q_d)
    model.quadrotor.default_plot(show=True)


if __name__ == "__main__":
    main()
