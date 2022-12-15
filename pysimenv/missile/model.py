from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.base import SimObject, DynSystem, ArrayType
from pysimenv.common.model import FirstOrderLinSys


class PlanarKin(DynSystem):
    def __init__(self, p_0: ArrayType, v_0: ArrayType, name="kin", **kwargs):
        super(PlanarKin, self).__init__(initial_states={'p': p_0, 'v': v_0}, name=name, **kwargs)

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
    def __init__(self, x_0: ArrayType, V, L_alp, L_delta, M_alp, M_q, M_delta, name="pitch_dyn", **kwargs):
        # x = [alp, theta, q]
        super(PitchDyn, self).__init__(initial_states={'x': x_0}, name=name, **kwargs)
        self.V = V
        self.L_alp = L_alp
        self.L_delta = L_delta
        self.M_alp = M_alp
        self.M_q = M_q
        self.M_delta = M_delta

    # implement
    def _deriv(self, x, delta: float):
        alp, q = x[0], x[2]

        alp_dot = q - 1./self.V*(self.L_alp*alp + self.L_delta*delta)
        theta_dot = q
        q_dot = self.M_alp*alp + self.M_q*q + self.M_delta*delta

        return {'x': np.array([alp_dot, theta_dot, q_dot])}

    # implement
    def _output(self):
        return self.state('x')

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

    def lift_accel(self, delta):
        return self.L_alp*self.alp + self.L_delta*delta

    def body_to_vel(self, v: np.ndarray) -> np.ndarray:
        return self.R_vb.dot(v)

    def plot(self, show=False):
        t = self.history('t')
        data = np.hstack((
            np.rad2deg(self.history('x')),
            np.expand_dims(np.rad2deg(self.history('delta')), axis=1)
        ))

        y_labels = ['alpha (deg)', 'theta (deg)', 'q (deg/s)', 'delta (deg)']

        fig, ax = plt.subplots(4, 1)
        for i in range(4):
            ax[i].plot(t, data[:, i])
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
    def __init__(self, kin: PlanarKin, name="vehicle", **kwargs):
        super(PlanarVehicle, self).__init__(name=name, **kwargs)
        self.kin = kin
        self._add_sim_objs([self.kin])

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
        if fig_ax:
            fig = fig_ax['fig']
            ax = fig_ax['ax']
        else:
            fig, ax = plt.subplots()
            ax.set_xlabel("p_x (m)")
            ax.set_ylabel("p_y (m)")
            ax.set_aspect('equal')
            ax.set_title("Flight Path")
            ax.grid()

        p = self.kin.history('p')

        if np.std(p[:, 0]) < 1e-2 and np.std(p[:, 1]) < 1e-2:
            ax.plot(p[0, 0], p[0, 1], marker='o', label=label)
        else:
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

    def __init__(self, p_0: ArrayType, V_0: float, gamma_0: float, acc_limit=np.array([-np.inf, np.inf]),
                 name="missile", **kwargs):
        v_0 = np.array([V_0*np.cos(gamma_0), V_0*np.sin(gamma_0)])
        kin = PlanarKin(p_0=p_0, v_0=v_0)
        super(PlanarMissile, self).__init__(kin=kin, name=name, **kwargs)
        self.fov_limit = np.inf  # Field-of-view limit
        self.acc_limit = acc_limit  # Acceleration limit
        self.ground_elev = -np.inf  # Ground elevation

    def look_angle(self, lam: float):
        sigma = self.kin.gamma - lam
        return sigma

    # implement
    def _forward(self, a_M_cmd: float):
        a_M = np.clip(a_M_cmd, self.acc_limit[0], self.acc_limit[1])
        a = self.kin.vel_to_inertial(np.array([0., a_M]))
        self.kin.forward(a=a)
        self._logger.append(t=self.time, V=self.V, gamma=self.gamma, a_M_cmd=a_M_cmd, a_M=a_M)

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
            self.name, self.p[0], self.p[1], self.V, np.rad2deg(self.gamma)))

        if self.flag == self.FLAG_OPERATING:
            print("[{:s}] Status: operating \n".format(self.name))
        elif self.flag == self.FLAG_STALLED:
            print("[{:s}] Status: stalled \n".format(self.name))
        elif self.flag == self.FLAG_COLLIDED:
            print("[{:s}] Status: collided \n".format(self.name))


class PlanarMissileWithPitch(PlanarMissile):
    def __init__(self, p_0: ArrayType, V_0: float, gamma_0: float,
                 pitch_dyn: PitchDyn, pitch_ap: SimObject, tau, name="missile", **kwargs):
        super(PlanarMissileWithPitch, self).__init__(p_0, V_0, gamma_0, name=name, **kwargs)
        self.pitch_dyn = pitch_dyn
        self.pitch_ap = pitch_ap
        self.act_dyn = FirstOrderLinSys(x_0=np.array([0.]), tau=tau)
        self._add_sim_objs([self.pitch_dyn, self.act_dyn, self.pitch_ap])

    # override
    def look_angle(self, lam: float):
        sigma = self.pitch_dyn.theta - lam
        return sigma

    def _forward(self, a_M_cmd: float):
        a_M_cmd = np.clip(a_M_cmd, self.acc_limit[0], self.acc_limit[1])
        
        delta = self.act_dyn.output
        q = self.pitch_dyn.output[2]
        a_L = self.pitch_dyn.lift_accel(delta)
        a_L_c = a_M_cmd  # approximation
        delta_c = self.pitch_ap.forward(q=q, a_L=a_L, a_L_c=a_L_c)

        self.act_dyn.forward(u=np.array([delta_c]))
        self.pitch_dyn.forward(delta=delta)

        a = self.pitch_dyn.body_to_vel(np.array([0., a_L]))
        a = self.kin.vel_to_inertial(a)
        self.kin.forward(a=a)
        self._logger.append(t=self.time, V=self.V, gamma=self.gamma, a_M_cmd=a_M_cmd, a_M=a_L)


class PlanarMovingTarget(PlanarVehicle):
    def __init__(self, p_0: ArrayType, V_0: float, gamma_0: float, name="target", **kwargs):
        v_0 = np.array([V_0*np.cos(gamma_0), V_0*np.sin(gamma_0)])
        kin = PlanarKin(p_0=p_0, v_0=v_0)
        super(PlanarMovingTarget, self).__init__(kin=kin, name=name, **kwargs)

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
