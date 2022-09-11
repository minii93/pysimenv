from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.base import SimObject, DynSystem, ArrayType
from pysimenv.common.model import FirstOrderLinSys


class PlanarKin(DynSystem):
    def __init__(self, p_0: ArrayType, v_0: ArrayType):
        super(PlanarKin, self).__init__(initial_states={'p': p_0, 'v': v_0}, name="kin")

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
        super(PitchDyn, self).__init__(initial_states={'x': x_0}, name="pitch_dyn")
        self.L_alp = L_alp
        self.L_delta = L_delta
        self.M_alp = M_alp
        self.M_q = M_q
        self.M_delta = M_delta

    # implement
    def _deriv(self, x, V: float, delta: float):
        alp, q = x[0], x[2]

        alp_dot = q - (self.L_alp*alp + self.L_delta*delta) / V
        theta_dot = q
        q_dot = self.M_alp*alp + self.M_q*q + self.M_delta*delta
        return {'x': np.array([alp_dot, theta_dot, q_dot])}


class PitchDynLatAccel(DynSystem):
    """
    x = [alp, theta, q]
    """
    def __init__(self, x_0: ArrayType, L_alp, L_delta, M_alp, M_q, M_delta):
        super(PitchDynLatAccel, self).__init__(initial_states={'x': x_0}, name="pitch_dyn")
        self.L_alp = L_alp
        self.L_delta = L_delta
        self.M_alp = M_alp
        self.M_q = M_q
        self.M_delta = M_delta

    # implement
    def _deriv(self, x, V: float, a_L: float):
        alp, q = x[0], x[2]

        alp_dot = q - 1. / V * a_L * np.cos(alp)
        theta_dot = q
        q_dot = (self.M_alp - self.M_delta * self.L_alp / self.L_delta) * alp + self.M_q * q + self.M_delta / self.L_delta * a_L
        return {'x': np.array([alp_dot, theta_dot, q_dot])}

    @property
    def alp(self) -> float:
        return self.state('x')[0]

    @property
    def theta(self) -> float:
        return self.state('x')[1]

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

    def plot(self, show=False):
        t = self.history('t')
        x = np.rad2deg(self.history('x'))
        y_labels = ['alpha (deg)', 'theta (deg)', 'q (deg/s)']

        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[i].plot(t, x[:, i])
            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel(y_labels[i])
            ax[i].grid()
        fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)


class PlanarVehicle(SimObject):
    def __init__(self, kin: PlanarKin, name="vehicle"):
        super(PlanarVehicle, self).__init__(name=name)
        self.kin = kin
        self._attach_sim_objs([self.kin])

    @property
    def p(self) -> np.ndarray:
        return self.kin.p

    @property
    def v(self) -> np.ndarray:
        return self.kin.v

    @property
    def V(self) -> float:
        return self.kin.V

    @property
    def gamma(self) -> float:
        return self.kin.gamma

    def _forward(self, *args, **kwargs) -> Union[None, np.ndarray, dict]:
        pass

    def plot_path(self, fig_ax=None, label="vehicle", show=False):
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
    FLAG_STALLED = 1
    FLAG_COLLIDED = 2

    def __init__(self, p_0: ArrayType, V_0: float, gamma_0: float, name="missile"):
        v_0 = np.array([V_0*np.cos(gamma_0), V_0*np.sin(gamma_0)])
        kin = PlanarKin(p_0=p_0, v_0=v_0)
        super(PlanarMissile, self).__init__(kin=kin, name=name)
        self.fov_limit = np.inf  # Field-of-view limit
        self.acc_limit = np.array([-np.inf, np.inf])  # Acceleration limit
        self.ground_elev = -np.inf  # Ground elevation

    def look_angle(self, lam: float):
        sigma = self.kin.gamma - lam
        return sigma

    # implement
    def _forward(self, a_M_cmd: float):
        a = self.kin.vel_to_inertial(np.array([0., a_M_cmd]))
        self.kin.forward(a=a)
        self._logger.append(t=self.time, V=self.V, gamma=self.gamma)

    # implement
    def _check_stop_condition(self) -> bool:
        to_stop = False
        if self.is_stalled():
            to_stop = True
            self.flag = self.FLAG_STALLED
        if self.is_collided():
            to_stop = True
            self.flag = self.FLAG_STALLED

        return to_stop

    def is_stalled(self):
        return self.kin.V < 1.  # when the speed is less than 1m/s

    def is_collided(self):
        return self.kin.p[1] < self.ground_elev

    def is_out_of_fov(self, lam: float):
        sigma = self.look_angle(lam)
        return abs(sigma) > self.fov_limit

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

    def report(self):
        np.set_printoptions(precision=2, suppress=True)

        print("[{:s}] Position: {:.2f}(m), {:.2f}(m), Speed: {:.2f}(m/s), Flight path angle: {:.2f}(deg)".format(
            self.name, self.p[0], self.p[1], self.V, np.deg2rad(self.gamma)))

        if self.flag == self.FLAG_OPERATING:
            print("[{:s}] Status: operating \n".format(self.name))
        elif self.flag == self.FLAG_STALLED:
            print("[{:s}] Status: stalled \n".format(self.name))
        elif self.flag == self.FLAG_COLLIDED:
            print("[{:s}] Status: collided \n".format(self.name))


class PlanarMissileWithPitch(PlanarMissile):
    def __init__(self, p_0: ArrayType, V_0: float, gamma_0: float,
                 pitch_dyn: PitchDynLatAccel, tau=0.5, name="missile"):
        super(PlanarMissileWithPitch, self).__init__(p_0, V_0, gamma_0, name=name)
        self.pitch_dyn = pitch_dyn
        self.accel_dyn = FirstOrderLinSys(x_0=np.array([0.]), tau=tau)
        self._attach_sim_objs([self.pitch_dyn, self.accel_dyn])

    # override
    def look_angle(self, lam: float):
        sigma = self.pitch_dyn.theta - lam
        return sigma

    def _forward(self, a_M_cmd: float):
        a_L_cmd = a_M_cmd/np.cos(self.pitch_dyn.alp)
        a_L = self.accel_dyn.forward(u=np.array([a_L_cmd]))
        self.pitch_dyn.forward(V=self.V, a_L=a_L)

        a = self.pitch_dyn.body_to_vel(np.array([0., a_L]))
        a = self.kin.vel_to_inertial(a)
        self.kin.forward(a=a)
        self._logger.append(t=self.time, V=self.V, gamma=self.gamma)


class PlanarMovingTarget(PlanarVehicle):
    def __init__(self, p_0: ArrayType, V_0: float, gamma_0: float, name="target"):
        v_0 = np.array([V_0*np.cos(gamma_0), V_0*np.sin(gamma_0)])
        kin = PlanarKin(p_0=p_0, v_0=v_0)
        super(PlanarMovingTarget, self).__init__(kin=kin, name=name)

    # implement
    @property
    def p(self):
        return self.kin.p

    @property
    def v(self):
        return self.kin.v

    def _forward(self):
        self.kin.forward(a=np.zeros(2))
        self._logger.append(t=self.time, V=self.kin.V, gamma=self.kin.gamma)
