import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from pysimenv.core.system import MultiStateDynSystem
from pysimenv.core.simulator import Simulator
from pysimenv.common.model import FlatEarthEnv
from pysimenv.common import orientation


class QuadrotorDynModel(MultiStateDynSystem):
    e3 = np.array([0., 0., 1.], dtype=np.float32)

    def __init__(self, initial_states: Union[list, tuple], m: float, J: np.ndarray):
        """
        :param initial_states: [p, v, R, omega] where
            p: position, (3,) numpy array
            v: velocity, (3,) numpy array
            R: rotation matrix, (3, 3) numpy array
            omega: angular velocity, (3,) numpy array
        :param m: mass, float
        :param J: inertia of mass, (3, 3) numpy array
        """
        super(QuadrotorDynModel, self).__init__(initial_states)
        self.m = m
        self.J = J
        self.grav_accel = FlatEarthEnv.grav_accel*self.e3

        self.state_var_list[2].attach_correction_fun(orientation.correct_orthogonality)

    # override
    def derivative(self, p, v, R, omega, u):
        """
        :param p: position, (3,) numpy array
        :param v: velocity, (3,) numpy array
        :param R: rotation matrix, (3, 3) numpy array
        :param omega: angular velocity, (3,) numpy array
        :param u: control input u = [f, tau], (4,) numpy array
        :return: [p_dot, v_dot, R_dot, omega_dot]
        """
        f = u[0]
        tau = u[1:4]

        p_dot = v
        v_dot = self.grav_accel - 1./self.m*(f*np.dot(R, self.e3))
        R_dot = np.matmul(R, self.hat(omega))
        omega_dot = np.linalg.solve(self.J, -np.cross(omega, np.dot(self.J, omega)) + tau)

        return [p_dot, v_dot, R_dot, omega_dot]

    @classmethod
    def hat(cls, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def plot_euler_angles(self):
        time_list = self.history('t')
        rotation_list = self.history('x_2')

        data_num = rotation_list.shape[0]
        euler_angle_list = np.zeros((data_num, 3))

        for i in range(data_num):
            R_bi = np.transpose(rotation_list[i, :, :])
            euler_angle_list[i, :] = orientation.rotation_to_euler_angles(R_bi)

        fig = plt.figure()
        names = ['phi', 'theta', 'psi']
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(time_list, np.rad2deg(euler_angle_list[:, i]), label="Actual")
            plt.xlabel("Time [s]")
            plt.ylabel(names[i])
            plt.grid()
            plt.legend()
        plt.suptitle("Euler angles")

        plt.draw()
        plt.pause(0.01)
        return fig


class QuadXThrustModel(object):
    def __init__(self, d_phi: float, d_theta: float, c_tau_f: float):
        self.d_phi = d_phi
        self.d_theta = d_theta
        self.c_tau_f = c_tau_f
        self.R_u = np.array([
            [1., 1., 1., 1.],
            [-d_phi/2., d_phi/2., d_phi/2., -d_phi/2.],
            [d_theta/2., -d_theta/2., d_theta/2., -d_theta/2.],
            [c_tau_f, c_tau_f, -c_tau_f, -c_tau_f]
        ])  # mapping matrix

    def convert(self, f_s: np.ndarray) -> np.ndarray:
        u = self.R_u.dot(f_s)
        return u


class QuadXMixer(object):
    def __init__(self, d_phi: float, d_theta: float, c_tau_f: float):
        self.d_phi = d_phi
        self.d_theta = d_theta
        self.c_tau_f = c_tau_f
        self.R_u = np.array([
            [1., 1., 1., 1.],
            [-d_phi/2., d_phi/2., d_phi/2., -d_phi/2.],
            [d_theta/2., -d_theta/2., d_theta/2., -d_theta/2.],
            [c_tau_f, c_tau_f, -c_tau_f, -c_tau_f]
        ])  # mapping matrix
        self.R_u_inv = np.linalg.inv(self.R_u)

    def convert(self, u_d: np.ndarray) -> np.ndarray:
        f_s = self.R_u_inv.dot(u_d)
        return f_s


def main():
    print("== Test for QuadrotorDynModel ==")
    m = 4.34
    J = np.diag([0.0820, 0.0845, 0.1377])

    pos = np.array([0., 0., -1.])
    vel = np.array([1., 0., 0.])
    R = np.identity(3)
    omega = np.array([0., 0., 0.1])
    initial_states_ = (pos, vel, R, omega)

    quadrotor = QuadrotorDynModel(initial_states_, m, J)

    u = np.array([45., 0., 0., 0.])

    simulator = Simulator(quadrotor)
    simulator.propagate(dt=0.01, time=10., save_history=True, u=u)

    quadrotor.default_plot()
    quadrotor.plot_euler_angles()
    plt.show()


if __name__ == "__main__":
    main()

