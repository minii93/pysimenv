import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint, BFGS, SR1
from pysimenv.common.model import FlatEarthEnv


class RelaxedHoverSolver(object):
    def __init__(self, m: float, J: np.ndarray, G: np.ndarray, kappa_f: float, kappa_tau: float,
                 fault_ind: list):
        self.m = m
        self.J = J
        self.G = G  # control effectiveness matrix
        self.kappa_f = kappa_f  # propeller coefficient
        self.kappa_tau = kappa_tau  # propeller coefficient

        self.num_motor = G.shape[1]
        self.fault_ind = fault_ind

    def _ravel(self, x: np.ndarray):
        """
        :param x: [omega_mag, p, q, r, r_ij, f_i]
        where omega = [p, q, r], R_0 = (r_ij), f = (f_i)
        :return:
        """
        omega_mag = x[0]
        omega = x[1:4]
        R_0 = x[4:13].reshape((3, 3))
        f = x[13:13 + self.num_motor]
        return omega_mag, omega, R_0, f

    def _unravel(self, omega_mag: float, omega: np.ndarray, R_0: np.ndarray, f: np.ndarray):
        x = np.hstack((np.array([omega_mag]), omega, R_0.flatten(), f))
        return x

    def constraint(self, x: np.ndarray):
        """
        :param omega_mag: magnitude of the angular velocity
        :param omega: [p, q, r] angular velocity expressed in the body coordinate
        :param R_0: rotation matrix from the body frame to the inertial frame
        :param f: [f_1, f_2, .., f_p] the thrust of each motor
        :return: constraints (=0)
        """
        omega_mag, omega, R_0, f = self._ravel(x)

        omega_i = np.array([0., 0., -omega_mag])
        z_B_i = R_0[:, 2]
        u = self.G.dot(f)
        F_l, tau = u[0], u[1:4]

        c_1 = (np.matmul(R_0, R_0.transpose()) - np.identity(3)).flatten()  # orthogonality condition
        c_2 = R_0.dot(omega) - omega_i  # angular velocity relation
        c_3 = np.array([
            np.dot(z_B_i, omega_i)*F_l/self.m - FlatEarthEnv.grav_accel*omega_mag
        ])  # total thrust condition
        c_4 = np.linalg.solve(self.J, -np.cross(omega, self.J.dot(omega)) + tau)  # zero angular acceleration
        c_5 = f[self.fault_ind]  # actuator fault condition
        return np.hstack((c_1, c_2, c_3, c_4, c_5))

    def power(self, x: np.ndarray):
        f = x[13:13 + self.num_motor]

        omega_p = np.sqrt(f/self.kappa_f)
        tau_p = self.kappa_tau*np.square(omega_p)
        return np.sum(tau_p*omega_p)

    def minimum_power_solution(self):
        # initial conditions
        omega_mag = 0.
        omega = np.zeros(3)
        R_0 = np.identity(3)
        f = np.ones(self.num_motor)*self.m*FlatEarthEnv.grav_accel/self.num_motor
        x_0 = self._unravel(omega_mag, omega, R_0, f)

        nl_con = NonlinearConstraint(self.constraint, 0., 0., jac="2-point", hess=BFGS())
        res = scipy.optimize.minimize(self.power, x_0, method='trust-constr', jac="2-point", hess=SR1(),
                                    constraints=nl_con, options={'verbose': 1})
        print(res.x)




