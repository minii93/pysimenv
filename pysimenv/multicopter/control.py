import numpy as np
import scipy
from pyquaternion import Quaternion
from typing import Union, Tuple
from pysimenv.core.base import BaseObject
from pysimenv.common.model import FlatEarthEnv
from pysimenv.common.orientation import quaternion_to_axis_angle


class FLVelControl(BaseObject):
    """
    Feedback linearization velocity control
    Reference: H. Voos, "Nonlinear Control of a Quadrotor Micro-UAV using Feedback-Linearization",
    Proceedings of the 2009 IEEE International Conference on Mechatronics, 2009.
    """
    def __init__(self, m: float, J: np.ndarray,
                 k_p_att: np.ndarray, k_d_att: np.ndarray, k_p_vel: np.ndarray,
                 interval: Union[int, float] = -1):
        super(FLVelControl, self).__init__(interval=interval)
        self.m = m
        self.J = J  # (3, 3) array
        self.k_p_att = k_p_att  # (3,) array
        self.k_d_att = k_d_att  # (3,) array
        self.k_p_vel = k_p_vel  # (3,) array

    # implement
    def evaluate(self, v: np.ndarray, eta: np.ndarray, omega: np.ndarray, v_d: np.ndarray = np.zeros(3)):
        """
        :param v: velocity
        :param eta:
        :param omega:
        :param v_d:
        :return:
        """
        u_tilde = self.k_p_vel*(v_d - v)
        u_t_1, u_t_2, u_t_3 = u_tilde[:]
        g = FlatEarthEnv.grav_accel

        f = self.m*np.sqrt(u_t_1**2 + u_t_2**2 + (u_t_3 - g)**2)
        phi_d = np.arcsin(self.m/f*u_t_2)
        theta_d = np.arctan(u_t_1/(u_t_3 - g))
        psi_d = 0.

        eta_d = np.array([phi_d, theta_d, psi_d])
        u_star = self.k_p_att*(eta_d - eta) - self.k_d_att*omega

        J_x, J_y, J_z = np.diag(self.J)[:]
        J_1 = (J_y - J_z)/J_x
        J_2 = (J_z - J_x)/J_y
        J_3 = (J_x - J_y)/J_z

        p, q, r = omega[:]
        tau_x = J_x*(-q*r*J_1 + u_star[0])
        tau_y = J_y*(-p*r*J_2 + u_star[1])
        tau_z = J_z*(-p*q*J_3 + u_star[2])

        u = np.array([f, tau_x, tau_y, tau_z])
        return u


class QuaternionAttControl(BaseObject):
    """
    Attitude control based on quaternion
    Reference: J. Carino, H. Abaunza and P. Castillo,
    "Quadrotor Quaternion Control",
    2015 International Conference on Unmanned Aircraft Systems (ICUAS), 2015.
    """
    def __init__(self, J: np.ndarray, K: np.ndarray, interval: Union[int, float] = -1):
        super(QuaternionAttControl, self).__init__(interval=interval)
        self.J = J  # (3, 3) array
        self.K = K  # (3, 6) array

    # implement
    def evaluate(self, q: np.ndarray, omega: np.ndarray, q_d: np.ndarray, omega_d: np.ndarray = np.zeros(3))\
            -> np.ndarray:
        a, phi = quaternion_to_axis_angle(q)
        theta = a*phi
        x = np.hstack((theta, omega))

        a_d, phi_d = quaternion_to_axis_angle(q_d)
        theta_d = a_d*phi_d
        x_d = np.hstack((theta_d, omega_d))

        u = -self.K.dot(x - x_d)
        tau = self.J.dot(u) + np.cross(omega, self.J.dot(omega))
        return tau

    @classmethod
    def gain(cls, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.identity(3)

        B = np.zeros((6, 3))
        B[3:6, :] = np.identity(3)

        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R).dot(B.transpose().dot(P))
        return K


class QuaternionPosControl(BaseObject):
    e3 = np.array([0., 0., 1.])
    n = np.array([0., 0., -1.])

    def __init__(self, m: float, K: np.ndarray, interval: Union[int, float] = -1):
        super(QuaternionPosControl, self).__init__(interval=interval)
        self.m = m
        self.K = K

    # implement
    def evaluate(self, p: np.ndarray, v: np.ndarray, p_d: np.ndarray, v_d: np.ndarray = np.zeros(3))\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.hstack((p, v))
        x_d = np.hstack((p_d, v_d))
        u = -self.K.dot(x - x_d)

        u_pd = u - FlatEarthEnv.grav_accel*self.e3
        f = self.m*np.linalg.norm(u_pd)

        r_d = np.hstack((
            np.array([self.n.dot(u_pd) + np.linalg.norm(u_pd)]),
            np.cross(self.n, u_pd)
        ))

        q_d = r_d/np.linalg.norm(r_d)

        x_dot = np.hstack((v, u))
        u_pd_dot = -self.K.dot(x_dot)
        r_d_dot = np.hstack((
            np.array([self.n.dot(u_pd_dot) + u_pd.dot(u_pd_dot)/np.linalg.norm(u_pd)]),
            np.cross(self.n, u_pd_dot)
        ))
        q_d_dot = r_d_dot/np.linalg.norm(r_d) + r_d*(
            -r_d.dot(r_d_dot)/np.linalg.norm(r_d)**3)

        omega_d = 2*Quaternion(q_d).conjugate*Quaternion(q_d_dot)
        omega_d = omega_d.elements[1:4]
        return f, q_d, omega_d

    @classmethod
    def gain(cls, Q: np.ndarray, R: np.ndarray):
        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.identity(3)

        B = np.zeros((6, 3))
        B[3:6, :] = np.identity(3)

        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R).dot(B.transpose().dot(P))
        return K
