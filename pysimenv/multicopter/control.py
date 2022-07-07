import numpy as np
from typing import Union
from pysimenv.core.base import SimObject
from pysimenv.common.model import FlatEarthEnv


class FLVelControl(SimObject):
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
    def forward(self, v: np.ndarray, eta: np.ndarray, omega: np.ndarray, v_d: np.ndarray = np.zeros(3)):
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
