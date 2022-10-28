import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from pysimenv.core.base import SimObject, DynSystem
from pysimenv.core.simulator import Simulator
from pysimenv.common.model import FlatEarthEnv
from pysimenv.common import orientation


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

    def plot_euler_angles(self):
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

        plt.draw()
        plt.pause(0.01)
        return fig


class QuadBase(SimObject):
    def __init__(self, dyn: MulticopterDyn, control: SimObject, effector: EffectorModel = None, mixer: Mixer = None, **kwargs):
        super(QuadBase, self).__init__(**kwargs)
        self.dyn = dyn
        self.control = control
        self.effector = effector
        self.mixer = mixer
        self._add_sim_objs([self.dyn, self.control])

    # implementation
    def _forward(self, **kwargs):
        """
        :param sigma_d: [x_d, y_d, z_d, phi_d]
        :param sigma_d_dot: [x_d_dot, y_d_dot, z_d_dot, phi_d_dot]
        :return:
        """
        if self.effector:
            if self.mixer:
                u_cmd = self.control.forward(self.dyn, **kwargs)  # u_cmd = {'f': total force, 'tau': moments}
                thrusts = self.mixer.convert(**u_cmd)
            else:
                thrusts = self.control.forward(self.dyn, **kwargs)  # thrusts = [f_1, f_2, ..., f_m]
            u = self.effector.convert(thrusts)
        else:
            u = self.control.forward(self.dyn, **kwargs)

        self.dyn.forward(**u)

    # implementation
    def _output(self) -> dict:
        return self.dyn.output


class QuadXEffector(EffectorModel):
    def __init__(self, d_phi: float, d_theta: float, c_tau_f: float):
        self.d_phi = d_phi
        self.d_theta = d_theta
        self.c_tau_f = c_tau_f
        self.B = np.array([
            [1., 1., 1., 1.],
            [-d_phi/2., d_phi/2., d_phi/2., -d_phi/2.],
            [d_theta/2., -d_theta/2., d_theta/2., -d_theta/2.],
            [c_tau_f, c_tau_f, -c_tau_f, -c_tau_f]
        ])  # mapping matrix

    def convert(self, f: np.ndarray) -> dict:
        u = self.B.dot(f)
        return {'f': u[0], 'tau': u[1:4]}

# class QuadXThrustModel(object):
#     def __init__(self, d_phi: float, d_theta: float, c_tau_f: float):
#         self.d_phi = d_phi
#         self.d_theta = d_theta
#         self.c_tau_f = c_tau_f
#         self.R_u = np.array([
#             [1., 1., 1., 1.],
#             [-d_phi/2., d_phi/2., d_phi/2., -d_phi/2.],
#             [d_theta/2., -d_theta/2., d_theta/2., -d_theta/2.],
#             [c_tau_f, c_tau_f, -c_tau_f, -c_tau_f]
#         ])  # mapping matrix
#
#     def convert(self, f_s: np.ndarray) -> np.ndarray:
#         u = self.R_u.dot(f_s)
#         return u


class QuadXMixer(Mixer):
    def __init__(self, d_phi: float, d_theta: float, c_tau_f: float):
        self.d_phi = d_phi
        self.d_theta = d_theta
        self.c_tau_f = c_tau_f
        self.B = np.array([
            [1., 1., 1., 1.],
            [-d_phi/2., d_phi/2., d_phi/2., -d_phi/2.],
            [d_theta/2., -d_theta/2., d_theta/2., -d_theta/2.],
            [c_tau_f, c_tau_f, -c_tau_f, -c_tau_f]
        ])  # mapping matrix
        self.B_inv = np.linalg.inv(self.B)

    def convert(self, f: float, tau: np.ndarray) -> np.ndarray:
        f = self.B_inv.dot(np.array([f, tau[0], tau[1], tau[2]]))
        return f


# class QuadXMixer(object):
#     def __init__(self, d_phi: float, d_theta: float, c_tau_f: float):
#         self.d_phi = d_phi
#         self.d_theta = d_theta
#         self.c_tau_f = c_tau_f
#         self.R_u = np.array([
#             [1., 1., 1., 1.],
#             [-d_phi/2., d_phi/2., d_phi/2., -d_phi/2.],
#             [d_theta/2., -d_theta/2., d_theta/2., -d_theta/2.],
#             [c_tau_f, c_tau_f, -c_tau_f, -c_tau_f]
#         ])  # mapping matrix
#         self.R_u_inv = np.linalg.inv(self.R_u)
#
#     def convert(self, u_d: np.ndarray) -> np.ndarray:
#         f_s = self.R_u_inv.dot(u_d)
#         return f_s


class ActuatorFault(SimObject):
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

    # implement
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

    quadrotor = MulticopterDyn(initial_states, m, J)

    u = np.array([45., 0., 0., 0.])

    simulator = Simulator(quadrotor)
    simulator.propagate(dt=0.01, time=10., save_history=True, u=u)

    quadrotor.default_plot()
    quadrotor.plot_euler_angles()
    plt.show()


if __name__ == "__main__":
    main()

