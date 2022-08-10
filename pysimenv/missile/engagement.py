import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from pysimenv.core.system import MultipleSystem
from pysimenv.missile.model import PlanarManVehicle2dof
from pysimenv.missile.guidance import Guidance2dim
from pysimenv.missile.util import RelKin2dim, CloseDistCond, closest_instant, lin_interp


class Engagement2dim(MultipleSystem):
    INTERCEPTED = 1
    MISSILE_STOP = 2
    IS_OUT_OF_VIEW = 3

    def __init__(self, missile: PlanarManVehicle2dof, target: PlanarManVehicle2dof, guidance: Guidance2dim):
        super(Engagement2dim, self).__init__()
        self.missile = missile
        self.target = target
        self.guidance = guidance
        self.rel_kin = RelKin2dim()
        self.close_dist_cond = CloseDistCond(r_threshold=10.0)

        self.attach_sim_objects([self.missile, self.target, self.guidance])

    # override
    def reset(self):
        super(Engagement2dim, self).reset()
        self.close_dist_cond.reset()

    # implement
    def initialize(self):
        x_M = self.missile.state('x')
        x_T = self.target.state('x')
        self.rel_kin.evaluate(x_M, x_T)

    # implement
    def _forward(self):
        x_M = self.missile.state('x')
        x_T = self.target.state('x')
        self.rel_kin.evaluate(x_M, x_T)
        self.close_dist_cond.evaluate(self.rel_kin.r)

        a_y_cmd = self.guidance.forward(self.missile, self.target, self.rel_kin)
        self.missile.forward(a_M_cmd=np.array([0., a_y_cmd]))
        self.target.forward()

    # implement
    def check_stop_condition(self) -> Tuple[bool, int]:
        to_stop = False

        missile_stop, _ = self.missile.check_stop_condition()
        if self.intercepted():  # probable interception
            to_stop = True
            self.flag = self.INTERCEPTED

        if missile_stop:  # stop due to the missile
            to_stop = True
            self.flag = self.MISSILE_STOP

        return to_stop, self.flag

    def intercepted(self) -> bool:
        return self.close_dist_cond.check()

    def state_on_closest_instant(self):
        t = self.missile.history('t')
        x_M = self.missile.history('x')
        x_T = self.target.history('x')

        p_M = x_M[:, 0:2]
        p_T = x_T[:, 0:2]
        index_close, xi_close = closest_instant(p_M, p_T)

        t_close = lin_interp(t[index_close], t[index_close + 1], xi_close)
        x_M_close = lin_interp(x_M[index_close], x_M[index_close + 1], xi_close)
        x_T_close = lin_interp(x_T[index_close], x_T[index_close + 1], xi_close)

        return t_close, x_M_close, x_T_close

    def miss_distance(self) -> float:
        _, x_M_c, x_T_c = self.state_on_closest_instant()

        p_M_c = x_M_c[0:2]
        p_T_c = x_T_c[0:2]

        d_miss = np.linalg.norm(p_M_c - p_T_c)
        return d_miss

    def impact_angle(self) -> float:
        _, x_M_c, x_T_c = self.state_on_closest_instant()
        gamma_M_c = x_M_c[3]
        gamma_T_c = x_T_c[3]
        return gamma_M_c - gamma_T_c

    def impact_time(self) -> float:
        t_c, _, _ = self.state_on_closest_instant()
        return t_c

    def report(self):
        self.missile.report()
        if self.flag == self.INTERCEPTED:
            print("[engagement] The target has been intercepted!")
        else:
            print("[engagement] The target has been missed!")

        _, x_M_c, x_T_c = self.state_on_closest_instant()

        print("[engagement] Missile state on the closest instant: {:.2f}(m), {:.2f}(m), {:.2f}(m/s), {:.2f}(deg) ".
              format(x_M_c[0], x_M_c[1], x_M_c[2], np.rad2deg(x_M_c[3]))
              )
        print("[engagement] Target state on the closest instant: {:.2f}(m), {:.2f}(m), {:.2f}(m/s), {:.2f}(deg) \n".
              format(x_T_c[0], x_T_c[1], x_T_c[2], np.rad2deg(x_T_c[3]))
              )

    def report_miss_distance(self):
        d_miss = self.miss_distance()
        print("[engagement] Miss distance: {:.6f} (m)".format(d_miss))

    def report_impact_angle(self):
        gamma_imp = self.impact_angle()
        print("[engagement] Impact angle: {:.2f} (deg)".format(np.rad2deg(gamma_imp)))

    def report_impact_time(self):
        t_imp = self.impact_time()
        print("[engagement] Impact time: {:.2f} (s)".format(t_imp))

    def plot_path(self):
        fig_axs = dict()

        fig_ax = self.missile.plot_path()
        self.target.plot_path(fig_ax)

        fig1, ax1 = fig_ax['fig'], fig_ax['ax']
        fig1.suptitle("2-dim flight path")
        fig_axs['path'] = {'fig': fig1, 'ax': ax1}

        return fig_axs

    def plot_zem(self):
        fig_axs = dict()

        relKin = RelKin2dim()
        t = self.missile.history('t')
        x_M = self.missile.history('x')
        x_T = self.target.history('x')

        zem = []
        for i in range(x_M.shape[0]):
            x_M_ = x_M[i, :]
            x_T_ = x_T[i, :]
            relKin.evaluate(x_M_, x_T_)

            zem.append(relKin.zem)
        zem = np.array(zem)

        fig, ax = plt.subplots()
        ax.set_title("ZEM")
        ax.plot(t[:-1], zem[:-1], label="ZEM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ZEM (m)")
        ax.grid()
        ax.legend()

        fig_axs['Rel. Kin. add.'] = {'fig': fig, 'ax': ax}
        return fig_axs

    def plot_rel_kin(self):
        fig_axs = dict()

        # plots for rel. kin.
        relKin = RelKin2dim()
        t = self.missile.history('t')
        x_M = self.missile.history('x')
        x_T = self.target.history('x')

        r = []
        sigma = []
        lam = []
        omega = []
        for i in range(x_M.shape[0]):
            x_M_ = x_M[i, :]
            x_T_ = x_T[i, :]
            relKin.evaluate(x_M_, x_T_)

            r.append(relKin.r)
            sigma.append(relKin.sigma)
            lam.append(relKin.lam)
            omega.append(relKin.omega)

        r = np.array(r)
        sigma = np.array(sigma)
        lam = np.array(lam)
        omega = np.array(omega)

        fig, ax = plt.subplots(4, 1, figsize=(6, 8))
        ax[0].set_title("Rel. dist")
        ax[0].plot(t[:-1], r[:-1], label="Rel. dist")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("r (m)")
        ax[0].grid()

        ax[1].set_title("Look angle")
        ax[1].plot(t[:-1], np.rad2deg(sigma[:-1]), label="look angle")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("sigma (deg)")
        ax[1].grid()

        ax[2].set_title("LOS angle")
        ax[2].plot(t[:-1], np.rad2deg(lam[:-1]), label="LOS angle")
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("lambda (deg)")
        ax[2].grid()

        ax[3].set_title("LOS rate")
        ax[3].plot(t[:-1], np.rad2deg(omega[:-1]), label="LOS rate")
        ax[3].set_xlabel("Time (s)")
        ax[3].set_ylabel("omega (deg/s)")
        ax[3].grid()
        fig.tight_layout()
        fig_axs['Rel. Kin.'] = {'fig': fig, 'ax': ax}

        return fig_axs


