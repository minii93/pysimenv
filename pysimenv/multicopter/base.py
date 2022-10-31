import numpy as np
import matplotlib.pyplot as plt

from pysimenv.core.base import DynSystem
from pysimenv.common.model import FlatEarthEnv
from pysimenv.common import orientation


class MulticopterDyn(DynSystem):
    e3 = np.array([0., 0., 1.], dtype=np.float32)

    def __init__(self, p_0: np.ndarray, v_0: np.ndarray, R_0: np.ndarray, omega_0: np.ndarray,
                 m: float, J: np.ndarray,
                 D_v: np.ndarray = np.zeros(3), D_omega: np.ndarray = np.zeros(3), **kwargs):
        """
        :param p_0: initial position, (3,) numpy array
            v_0: initial velocity, (3,) numpy array
            R_0: initial rotation, (3, 3) numpy array
            omega_0: initial angular velocity, (3,) numpy array
        :param m: mass, float
        :param J: inertia of mass, (3, 3) numpy array
        """
        super(MulticopterDyn, self).__init__(
            initial_states={'p': p_0, 'v': v_0, 'R': R_0, 'omega': omega_0}, **kwargs
        )
        self.m = m
        self.J = J
        self.D_v = D_v
        self.D_omega = D_omega
        self.grav_accel = FlatEarthEnv.grav_accel*self.e3

        self.state_vars['R'].attach_correction_fun(orientation.correct_orthogonality)

    # implementation
    def _deriv(self, p, v, R, omega, f: float, tau: np.ndarray):
        """
        :param p: position, (3,) numpy array
        :param v: velocity, (3,) numpy array
        :param R: rotation matrix, (3, 3) numpy array
        :param omega: angular velocity, (3,) numpy array
        :param f: total thrust, float
        :param tau: moments, (3,) numpy array
        :return: [p_dot, v_dot, R_dot, omega_dot]
        """
        p_dot = v.copy()
        v_dot = self.grav_accel - 1./self.m*(f*np.dot(R, self.e3)) - 1./self.m*self.D_v.dot(v)
        R_dot = np.matmul(R, self.hat(omega))
        omega_dot = np.linalg.solve(self.J, -np.cross(omega, np.dot(self.J, omega)) + tau - self.D_omega.dot(omega))

        return {'p': p_dot, 'v': v_dot, 'R': R_dot, 'omega': omega_dot}

    # implement
    def _output(self) -> dict:
        return self.state()

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
        # rotation matrix from the body frame to the inertial frame R_ib
        R_ib = self.state('R').copy()
        return R_ib

    @property
    def quaternion(self) -> np.ndarray:
        R_ib = self.state('R')
        q = orientation.rotation_to_quaternion(np.transpose(R_ib))
        return q

    @property
    def euler_ang(self) -> np.ndarray:
        R_ib = self.state('R')
        eta = np.array(
            orientation.rotation_to_euler_angles(np.transpose(R_ib))
        )
        return eta

    @property
    def ang_vel(self) -> np.ndarray:
        # angular velocity of the vehicle frame with respect to the inertial frame
        return self.state('omega')

    def plot_path(self, show: bool = False):
        p = self.history('p')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(p[:, 0], p[:, 1], -p[:, 2])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("h (m)")
        ax.grid()
        ax.set_title("Flight path")

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.1)

    def plot_euler_angles(self, show: bool = False):
        t = self.history('t')
        R = self.history('R')

        data_num = R.shape[0]
        euler_angle_list = np.zeros((data_num, 3))

        for i in range(data_num):
            R_bi = np.transpose(R[i, :, :])
            euler_angle_list[i, :] = orientation.rotation_to_euler_angles(R_bi)

        fig = plt.figure()
        names = ['phi', 'theta', 'psi']
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(t, np.rad2deg(euler_angle_list[:, i]), label="Actual")
            plt.xlabel("Time (s)")
            plt.ylabel(names[i])
            plt.grid()
            plt.legend()
        plt.suptitle("Euler angles")

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)
        return fig


class EffectorModel(object):
    # to be implemented
    def convert(self, f: np.ndarray) -> dict:
        """
        :param f: thrust of each motor [f_1, f_2, ..., f_m]
        :return: {'f': float, 'tau': np.ndarray} total thrust and moments
        """
        raise NotImplementedError


class Mixer(object):
    # to be implemented
    def convert(self, f: float, tau: np.ndarray) -> np.ndarray:
        """
        :param f: total thrust
        :param tau: moments
        :return: thrust command of each motor [f_1, f_2, ..., f_m]
        """
        raise NotImplementedError


