import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from pysimenv.core.base import ArrayType
from pysimenv.core.system import DynSystem
from pysimenv.common.util import wrap_to_pi


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
        return self.state['x'][0:2]

    @property
    def v(self) -> np.ndarray:
        """
        :return: [v_x, v_y]
        """
        x = self.state['x']
        V = x[2]
        gamma = x[3]

        return np.array([V*np.cos(gamma), V*np.sin(gamma)])

    @property
    def V(self) -> float:
        """
        :return: V
        """
        return self.state['x'][2]

    @property
    def gamma(self) -> float:
        """
        :return: gamma
        """
        return self.state['x'][3]

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
    def check_stop_condition(self) -> Tuple[bool, int]:
        to_stop = False
        if self.is_stalled():
            to_stop = True
            self.flag = self.STALLED
        if self.is_collided():
            to_stop = True
            self.flag = self.COLLIDED
        return to_stop, self.flag

    def is_stalled(self) -> bool:
        return self.V < 10  # when the speed is less than 10m/s

    def is_collided(self) -> bool:
        return self.p[1] < self.ground_elevation - 0.5

    def is_out_of_view(self, sigma: float) -> bool:
        return abs(sigma) > self.fov_limit

    def report(self):
        np.set_printoptions(precision=2, suppress=True)
        x_M_f = self.state['x']

        print("[{:s}] Final state: {:.2f}(m), {:.2f}(m), {:.2f}(m/s), {:.2f}(deg)".format(
            self.name, x_M_f[0], x_M_f[1], x_M_f[2], np.rad2deg(x_M_f[3])))

        if self.flag == self.NORMAL:
            print("[{:s}] Status: normal \n".format(self.name))
        elif self.flag == self.STALLED:
            print("[{:s}] Status: stalled \n".format(self.name))
        elif self.flag == self.COLLIDED:
            print("[{:s}] Status: collided \n".format(self.name))
