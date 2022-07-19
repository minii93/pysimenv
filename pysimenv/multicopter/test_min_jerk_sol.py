import numpy as np
from pysimenv.core.system import MultipleSystem
from pysimenv.core.simulator import Simulator
from pysimenv.multicopter.model import MulticopterDynamic
from pysimenv.multicopter.control import QuaternionPosControl, QuaternionAttControl
from pysimenv.multicopter.planning import MinJerkSol


class Model(MultipleSystem):
    def __init__(self):
        super(Model, self).__init__()

        # Quadrotor dynamic model
        m = 1.3
        J = np.diag([0.0119, 0.0119, 0.0219])

        p_0 = np.array([-5., 0., 0.])
        v_0 = np.array([1., 0., 0.])
        R_iv_0 = np.identity(3)
        omega_0 = np.zeros(3)
        self.dynamic = MulticopterDynamic([p_0, v_0, R_iv_0, omega_0], m, J)

        # Controller
        K_att = QuaternionAttControl.gain(
            Q=np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2,
            R=1e-4*np.identity(3)
        )
        K_pos = QuaternionPosControl.gain(
            Q=np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2,
            R=np.identity(3)
        )
        self.att_control = QuaternionAttControl(J, K_att)
        self.pos_control = QuaternionPosControl(m, K_pos)

        # Trajectory
        t_0 = 0.
        t_f = 10.
        s_x_0 = np.array([p_0[0], v_0[0], 0.])
        s_y_0 = np.array([p_0[1], v_0[1], 0.])
        s_z_0 = np.array([p_0[2], v_0[2], 0.])
        self.x_trajectory = MinJerkSol(
            t_0=t_0, t_f=t_f, s_0=s_x_0, s_f=np.zeros(3)
        )
        self.y_trajectory = MinJerkSol(
            t_0=t_0, t_f=t_f, s_0=s_y_0, s_f=np.zeros(3)
        )
        self.z_trajectory = MinJerkSol(
            t_0=t_0, t_f=t_f, s_0=s_z_0, s_f=np.zeros(3)
        )
        self.t_f = t_f

        self.attach_sim_objects([
            self.dynamic, self.att_control, self.pos_control,
            self.x_trajectory, self.y_trajectory, self.z_trajectory
        ])

    def forward(self) -> None:
        # Trajectory
        if self.time < self.t_f:
            s_x = self.x_trajectory.forward()
            s_y = self.y_trajectory.forward()
            s_z = self.z_trajectory.forward()
            p_d = np.array([s_x[0], s_y[0], s_z[0]])
            v_d = np.array([s_x[1], s_y[1], s_z[1]])
        else:
            p_d = np.zeros(3)
            v_d = np.zeros(3)

        # Position and attitude control
        p = self.dynamic.pos
        v = self.dynamic.vel
        q = self.dynamic.quaternion
        omega = self.dynamic.ang_vel

        f, q_d, omega_d = self.pos_control.forward(p, v, p_d, v_d)
        tau = self.att_control.forward(q, omega, q_d, omega_d)

        self.dynamic.forward(np.array([f, tau[0], tau[1], tau[2]]))


def main():
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(0.01, 15., True)
    model.dynamic.default_plot(show=True)


if __name__ == "__main__":
    main()
