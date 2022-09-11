from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.base import SimObject, DynSystem, ArrayType
from pysimenv.common.util import wrap_to_pi


class PlanarKin(DynSystem):
    def __init__(self, p_0: ArrayType, v_0: ArrayType):
        super(PlanarKin, self).__init__(initial_states={'p': p_0, 'v': v_0})

    # implement
    def _deriv(self, p, v, a: np.ndarray):
        p_dot = v.copy()
        v_dot = a.copy()
        return {'p': p_dot, 'v': v_dot}

    @property
    def p(self) -> np.ndarray:
        return self.state('p').copy()

    @property
    def v(self) -> np.ndarray:
        return self.state('v').copy()

    @property
    def V(self) -> float:
        v = self.state('v')
        return np.linalg.norm(v)

    @property
    def gamma(self) -> float:
        v = self.state('v')
        return np.arctan2(v[1], v[0])

    @property
    def R_iv(self) -> np.ndarray:
        c_gamma = np.cos(self.gamma)
        s_gamma = np.sin(self.gamma)
        return np.array([
            [c_gamma, -s_gamma],
            [s_gamma, c_gamma]
        ])

    def vel_to_inertial(self, v: np.ndarray) -> np.ndarray:
        return self.R_iv.dot(v)

    def inertial_to_vel(self, v: np.ndarray) -> np.ndarray:
        return self.R_iv.transpose().dot(v)


class PitchDyn(DynSystem):
    def __init__(self, x_0: ArrayType, L_alp, L_delta, M_alp, M_q, M_delta):
        super(PitchDyn, self).__init__(initial_states={'x': x_0})
        self.L_alp = L_alp
        self.L_delta = L_delta
        self.M_alp = M_alp
        self.M_q = M_q
        self.M_delta = M_delta

    # implement
    def _deriv(self, x, V_M: float, delta: float):
        alp, q = x[0], x[2]
        alp_dot = q - (self.L_alp*alp + self.L_delta*delta)/V_M
        theta_dot = q
        q_dot = self.M_alp*alp + self.M_q*q + self.M_delta*delta
        return {'alp': alp_dot, 'theta': theta_dot, 'q': q_dot}


class PitchDynLatAccel(DynSystem):
    def __init__(self, x_0: ArrayType, L_alp, L_del, M_alp, M_q, M_del):
        super(PitchDynLatAccel, self).__init__(initial_states={'x': x_0})
        self.L_alp = L_alp
        self.L_del = L_del
        self.M_alp = M_alp
        self.M_q = M_q
        self.M_del = M_del

    # implement
    def _deriv(self, x, V_M: float, a_L: float):
        alp, q = x[0], x[2]

        alp_dot = q - 1./V_M*a_L*np.cos(alp)
        theta_dot = q
        q_dot = (self.M_alp - self.M_del*self.L_alp/self.L_del)*alp + self.M_q*q + self.M_del/self.L_del*a_L
        return {'alp': alp_dot, 'theta': theta_dot, 'q': q_dot}

    @property
    def alp(self) -> float:
        return self.state('x')[0]

    @property
    def R_vb(self) -> np.ndarray:
        c_alp = np.cos(self.alp)
        s_alp = np.sin(self.alp)
        return np.array([
            [c_alp, -s_alp],
            [s_alp, c_alp]
        ])

    def body_to_vel(self, v: np.ndarray) -> np.ndarray:
        return self.R_vb.dot(v)


class PlanarVehicle(SimObject):
    def __init__(self, kin: PlanarKin):
        super(PlanarVehicle, self).__init__()
        self.kin = kin
        self._attach_sim_objs([self.kin])

    # to be implemented
    @property
    def p(self) -> np.ndarray:
        raise NotImplementedError

    def v(self) -> np.ndarray:
        raise NotImplementedError

    def _forward(self, *args, **kwargs) -> Union[None, np.ndarray, dict]:
        raise NotImplementedError

    def plot_path(self, fig_ax=None, label='vehicle', show=False):
        if fig_ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel("p_x (m)")
            ax.set_ylabel("p_y (m)")
            ax.set_aspect('equal')
            ax.set_title("Flight Path")
            ax.grid()
        else:
            fig = fig_ax['fig']
            ax = fig_ax['ax']

        p = self.kin.history('p')
        ax.plot(p[:, 0], p[:, 1], label=label)
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)
        return {'fig': fig, 'ax': ax}


class PlanarMissile(PlanarVehicle):
    def __init__(self, p_0: ArrayType, V_0: float, gamma_0: float):
        v_0 = np.array([V_0*np.cos(gamma_0), V_0*np.sin(gamma_0)])
        kin = PlanarKin(p_0=p_0, v_0=v_0)
        super(PlanarMissile, self).__init__(kin=kin)

    # implement
    @property
    def p(self):
        return self.kin.p

    @property
    def v(self):
        return self.kin.v

    def _forward(self, a_M_cmd: float):
        a = self.kin.vel_to_inertial(np.array([0., a_M_cmd]))
        self.kin.forward(a=a)
        self._logger.append(V=self.kin.V, gamma=self.kin.gamma)

    def plot_kin(self, show=False):
        t = self.kin.history('t')
        data = [self.kin.history('p')[:, 0],
                self.kin.history('p')[:, 1],
                self.history('V'),
                np.rad2deg(self.history('gamma'))]
        labels = ['p_x', 'p_y', 'V', 'gamma']

        fig, ax = plt.subplots(4, 1)
        for i in range(4):
            ax[i].plot(t, data[i], label=labels[i])
            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel(labels[i])
            ax[i].grid()
        fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)


class PlanarMissileWithPitch(PlanarMissile):
    def __init__(self, p_0: ArrayType, V_0: float, gamma_0: float, pitch_dyn: PitchDynLatAccel):
        super(PlanarMissileWithPitch, self).__init__(p_0, V_0, gamma_0)
        self.pitch_dyn = pitch_dyn
        self._attach_sim_objs([self.pitch_dyn])

    def _forward(self, a_M_cmd: float):
        a_L = a_M_cmd/np.cos(self.pitch_dyn.alp)
        self.pitch_dyn.forward(a_L=a_L)

        a = self.pitch_dyn.body_to_vel(a_L)
        a = self.kin.vel_to_inertial(a)
        self.kin.forward(a=a)


class PlanarMovingTarget(PlanarVehicle):
    def __init__(self, p_0: ArrayType, V_0: float, gamma_0: float):
        v_0 = np.array([V_0*np.cos(gamma_0), V_0*np.sin(gamma_0)])
        kin = PlanarKin(p_0=p_0, v_0=v_0)
        super(PlanarMovingTarget, self).__init__(kin=kin)
        self._attach_sim_objs([self.kin])

    # implement
    @property
    def p(self):
        return self.kin.p

    @property
    def v(self):
        return self.kin.v

    def _forward(self):
        self.kin.forward(a=np.zeros(2))


class PlanarManVehicle2dof(DynSystem):
    """
    A 2-dof model for a maneuvering vehicle in the plane
    state x: [p_x, p_y, V, gamma]
    control u: [a_x, a_y]
    where [p_x, p_y] is the position expressed in the inertial frame and
    [a_x, a_y] is the acceleration expressed in the velocity frame
    """
    def __init__(self, initial_state: ArrayType):
        super(PlanarManVehicle2dof, self).__init__(initial_states={'x': initial_state})
        self.name = "planar_man_vehicle"

        def angle_correction_fun(state: np.ndarray) -> np.ndarray:
            state[3] = wrap_to_pi(state[3])  # gamma should be in [-pi, pi] rad
            return state
        self.state_vars['x'].attach_correction_fun(angle_correction_fun)

    def _deriv(self, x: np.ndarray, u: np.ndarray = np.zeros(2)):
        """
        :param x: [p_x, p_y, V, gamma]
        :param u: [a_x, a_y]
        :return: x_dot
        """
        V = x[2]
        gamma = x[3]

        a_x = u[0]
        a_y = u[1]

        p_x_dot = V*np.cos(gamma)
        p_y_dot = V*np.sin(gamma)
        V_dot = a_x

        if abs(V) < 1e-4:
            gamma_dot = 0
        else:
            gamma_dot = a_y/V

        return {'x': np.array([p_x_dot, p_y_dot, V_dot, gamma_dot])}

    @property
    def p(self) -> np.ndarray:
        """
        :return: [p_x, p_y]
        """
        return self.state('x')[0:2]

    @property
    def v(self) -> np.ndarray:
        """
        :return: [v_x, v_y]
        """
        x = self.state('x')
        V = x[2]
        gamma = x[3]

        return np.array([V*np.cos(gamma), V*np.sin(gamma)])

    @property
    def V(self) -> float:
        """
        :return: V
        """
        return self.state('x')[2]

    @property
    def gamma(self) -> float:
        """
        :return: gamma
        """
        return self.state('x')[3]

    def plot(self):
        var_keys = {'x', 'u'}
        var_ind_dict = {'x': [0, 1, 2, 3], 'u': [0, 1]}
        var_names_dict = {
            'x': ['p_x (m)', 'p_y (m)', 'V (m/s)', 'gamma (rad)'],
            'u': ['a_x (m/s**2)', 'a_y (m/s**2)']
        }
        fig_axs = self.default_plot(var_keys=var_keys, var_ind_dict=var_ind_dict, var_names_dict=var_names_dict)
        return fig_axs

    def plot_path(self, fig_ax=None, show=False):
        if fig_ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel("p_x (m)")
            ax.set_ylabel("p_y (m)")
            ax.grid()
        else:
            fig = fig_ax['fig']
            ax = fig_ax['ax']

        x = self.history('x')
        p = x[:, 0:2]

        ax.plot(p[:, 0], p[:, 1], label=self.name + " path")
        ax.set_aspect('equal')
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)
        return {'fig': fig, 'ax': ax}


class PlanarNonManVehicle2dof(PlanarManVehicle2dof):
    def __init__(self, initial_state: ArrayType):
        super(PlanarNonManVehicle2dof, self).__init__(initial_state)
        self.name = "planar_non_man_vehicle"

    # override
    def plot(self):
        var_keys = {'x'}
        var_ind_dict = {'x': [0, 1, 2, 3]}
        var_names_dict = {
            'x': ['p_x (m)', 'p_y (m)', 'V (m/s)', 'gamma (rad)']
        }
        fig_axs = self.default_plot(var_keys, var_ind_dict, var_names_dict)
        return fig_axs

    # override
    def plot_path(self, fig_ax=None, show=False):
        if fig_ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel("p_x (m)")
            ax.set_ylabel("p_y (m)")
            ax.grid()
        else:
            fig = fig_ax['fig']
            ax = fig_ax['ax']

        x = self.history('x')
        p = x[:, 0:2]

        if abs(self.V) < 1e-4:
            ax.plot(p[0, 0], p[0, 1], marker='o', label=self.name)
        else:
            ax.plot(p[:, 0], p[:, 1], label=self.name + " path")
        ax.set_aspect('equal')
        ax.legend()
        fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)
        return {'fig': fig, 'ax': ax}


class PlanarMissile2dof(PlanarManVehicle2dof):
    NORMAL = 0
    STALLED = 1
    COLLIDED = 2

    def __init__(self, initial_state: ArrayType):
        super(PlanarMissile2dof, self).__init__(initial_state)
        self.name = "missile"
        self.fov_limit = np.inf
        self.acc_limit = np.array([
            [-np.inf, -np.inf],
            [np.inf, np.inf]
        ])
        self.ground_elevation = -np.inf

    def _forward(self, a_M_cmd: np.ndarray = np.zeros(2)):
        a_M = np.clip(a_M_cmd, self.acc_limit[0], self.acc_limit[1])
        super(PlanarMissile2dof, self)._forward(u=a_M)
        self._logger.append(a_M_cmd=a_M_cmd)

    # implement
    def _check_stop_condition(self) -> bool:
        to_stop = False
        if self.is_stalled():
            to_stop = True
            self.flag = self.STALLED
        if self.is_collided():
            to_stop = True
            self.flag = self.COLLIDED
        return to_stop

    def is_stalled(self) -> bool:
        return self.V < 10  # when the speed is less than 10m/s

    def is_collided(self) -> bool:
        return self.p[1] < self.ground_elevation - 0.5

    def is_out_of_view(self, sigma: float) -> bool:
        return abs(sigma) > self.fov_limit

    def report(self):
        np.set_printoptions(precision=2, suppress=True)
        x_M_f = self.state('x')

        print("[{:s}] Final state: {:.2f}(m), {:.2f}(m), {:.2f}(m/s), {:.2f}(deg)".format(
            self.name, x_M_f[0], x_M_f[1], x_M_f[2], np.rad2deg(x_M_f[3])))

        if self.flag == self.NORMAL:
            print("[{:s}] Status: normal \n".format(self.name))
        elif self.flag == self.STALLED:
            print("[{:s}] Status: stalled \n".format(self.name))
        elif self.flag == self.COLLIDED:
            print("[{:s}] Status: collided \n".format(self.name))
