import numpy as np
from typing import Union
from pysimenv.core.base import DynSystem
from pysimenv.common.model import FlatEarthEnv


class FixedTimeFaultEstimator(DynSystem):
    """
    Reference: L. Guo, "Fixed-Time Observer Based Safety Control for a Quadrotor UAV",
    IEEE Transactions on Aerospace and Electronic Systems, 2021.
    """
    def __init__(self, initial_states: Union[list, tuple],
                 alpha: float, beta: float, k_1: float, k_2: float,
                 m: float, J: np.ndarray, **kwargs):
        super(FixedTimeFaultEstimator, self).__init__(
            initial_states={'z_1': initial_states[0], 'z_2': initial_states[1]}, **kwargs
        )
        self.alpha = alpha
        self.beta = beta
        self.k_1 = k_1
        self.k_2 = k_2
        self.m = m
        self.J = J.copy()

    def _deriv(self, z_1: np.ndarray, z_2: np.ndarray, x: np.ndarray, eta: np.ndarray, u: np.ndarray):
        """
        :param z_1: estimation for the state
        :param z_2: estimation for the actuator fault
        :param x: actual state (v_z, p, q, r)
        :param eta: Euler angles (phi, theta, psi)
        :param f_s: Commanded thrust force
        :return:
        """
        phi, theta = eta[0], eta[1]
        p, q, r = x[1:4]
        J_x, J_y, J_z = self.J[0, 0], self.J[1, 1], self.J[2, 2]
        f = np.array([
            FlatEarthEnv.grav_accel,
            (J_y - J_z)/J_x*q*r,
            (J_z - J_x)/J_y*p*r,
            (J_x - J_y)/J_z*p*q
        ])
        B = np.diag([
            -np.cos(phi)*np.cos(theta)/self.m, 1./J_x, 1./J_y, 1./J_z
        ])

        e_d = z_1 - x
        z_1_dot = -self.k_1*(
            np.power(np.abs(e_d), self.alpha)*np.sign(e_d) + np.power(np.abs(e_d), self.beta)*np.sign(e_d)) +\
            f + z_2 + B.dot(u)
        z_2_dot = -self.k_2*(
            np.power(np.abs(e_d), 2*self.alpha - 1)*np.sign(e_d) + np.power(np.abs(e_d), 2*self.beta - 1)*np.sign(e_d)
        )
        return {'z_1': z_1_dot, 'z_2': z_2_dot}

    @property
    def delta_hat(self) -> np.ndarray:
        return self.state('z_2')
