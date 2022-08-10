import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from pysimenv.core.base import StaticObject
from pysimenv.core.system import DynSystem
from pysimenv.core.simulator import Simulator
from pysimenv.common.model import FlatEarthEnv
from pysimenv.common import orientation


class MulticopterDynamic(DynSystem):
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
        super(MulticopterDynamic, self).__init__(
            initial_states={
                'p': initial_states[0], 'v': initial_states[1],
                'R': initial_states[2], 'omega': initial_states[3]}
        )
        self.m = m
        self.J = J
        self.grav_accel = FlatEarthEnv.grav_accel*self.e3

        self.state_vars['R'].attach_correction_fun(orientation.correct_orthogonality)

    # implement
    def _deriv(self, p, v, R, omega, u):
        """
        :param p: position, (3,) numpy array
        :param v: velocity, (3,) numpy array
        :param R: rotation matrix, (3, 3) numpy array
        :param omega: angular velocity, (3,) numpy array
        :param u: control input u = [f, tau], (4,) numpy array
        :return: [p_dot, v_dot, R_dot, omega_dot]
        """
        f = u[0]  # Total thrust
        tau = u[1:4]  # Moments

        p_dot = v
        v_dot = self.grav_accel - 1./self.m*(f*np.dot(R, self.e3))
        R_dot = np.matmul(R, self.hat(omega))
        omega_dot = np.linalg.solve(self.J, -np.cross(omega, np.dot(self.J, omega)) + tau)

        return {'p': p_dot, 'v': v_dot, 'R': R_dot, 'omega': omega_dot}

    @classmethod
    def hat(cls, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @property
    def pos(self) -> np.ndarray:
        # position in the inertial frame
        return self.state('p').copy()

    @property
    def vel(self) -> np.ndarray:
        # velocity in the inertial frame
        return self.state('v').copy()

    @property
    def rotation(self) -> np.ndarray:
        # rotation matrix from the vehicle frame to the inertial frame R_iv
        R_iv = self.state('R').copy()
        return R_iv

    @property
    def quaternion(self) -> np.ndarray:
        R_iv = self.state('R')
        q = orientation.rotation_to_quaternion(np.transpose(R_iv))
        return q

    @property
    def euler_ang(self) -> np.ndarray:
        R_iv = self.state('R')
        eta = np.array(
            orientation.rotation_to_euler_angles(np.transpose(R_iv))
        )
        return eta

    @property
    def ang_vel(self) -> np.ndarray:
        # angular velocity of the vehicle frame with respect to the inertial frame
        return self.state('omega')

    def plot_euler_angles(self):
        time_list = self.history('t')
        rotation_list = self.history('R')

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
            plt.xlabel("Time (s)")
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


class ActuatorFault(StaticObject):
    def __init__(self, t_list: List[float], alp_list: List[np.ndarray], rho_list: List[np.ndarray],
                 interval: Union[int, float] = -1):
        """
        :param t_list: [t_0, t_1, ..., t_{k-1}] time of fault occurrence
        :param alp_list: [lam_0: (M,) array, ..., lam_{k-1}] gain fault, M: the number of motors
        :param rho_list: [rho_0: (M,) array, ..., rho_{k-1}] bias fault, M: the number of motors
        :return:
        """
        super(ActuatorFault, self).__init__(interval)
        assert len(t_list) == len(alp_list), "Array sizes doesn't match."
        assert len(t_list) == len(rho_list), "Array sizes doesn't match."
        self.t_list = t_list
        self.alp_list = alp_list
        self.rho_list = rho_list

        self.alp = alp_list[0]
        self.rho = rho_list[0]
        self.next_ind = 1

    def _forward(self, f_s: np.ndarray):
        if self.next_ind < len(self.t_list):
            if self.time >= self.t_list[self.next_ind]:
                self.alp = self.alp_list[self.next_ind]
                self.rho = self.rho_list[self.next_ind]
                self.next_ind += 1
        f_s_star = self.alp*f_s + self.rho
        return f_s_star


def main():
    print("== Test for QuadrotorDynModel ==")
    m = 4.34
    J = np.diag([0.0820, 0.0845, 0.1377])

    pos = np.array([0., 0., -1.])
    vel = np.array([1., 0., 0.])
    R = np.identity(3)
    omega = np.array([0., 0., 0.1])
    initial_states = (pos, vel, R, omega)

    quadrotor = MulticopterDynamic(initial_states, m, J)

    u = np.array([45., 0., 0., 0.])

    simulator = Simulator(quadrotor)
    simulator.propagate(dt=0.01, time=10., save_history=True, u=u)

    quadrotor.default_plot()
    quadrotor.plot_euler_angles()
    plt.show()


if __name__ == "__main__":
    main()

