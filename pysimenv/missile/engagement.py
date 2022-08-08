import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from pysimenv.core.system import MultipleSystem
from pysimenv.missile.model import PlanarMissile2dof, PlanarManVehicle2dof
from pysimenv.missile.guidance import PurePNG2dim, IACBPNG
from pysimenv.missile.util import RelKin2dim, CloseDistCond, miss_distance, closest_instant, lin_interp


class Engagement2dim(MultipleSystem):
    def __init__(self, missile: PlanarMissile2dof, target: PlanarManVehicle2dof):
        super(Engagement2dim, self).__init__()
        self.missile = missile
        self.target = target
        self.rel_kin = RelKin2dim()
        self.close_dist_cond = CloseDistCond(r_threshold=10.0)

        self.attach_sim_objects([self.missile, self.target])

    # override
    def reset(self):
        super(Engagement2dim, self).reset()
        self.close_dist_cond.reset()

    # implement
    def initialize(self):
        x_M = self.missile.state['x']
        x_T = self.target.state['x']
        self.rel_kin.evaluate(x_M, x_T)

    # implement
    def forward(self):
        x_M = self.missile.state['x']
        x_T = self.target.state['x']
        self.rel_kin.evaluate(x_M, x_T)
        self.close_dist_cond.evaluate(self.rel_kin.r)
        self.target.forward()

    # implement
    def check_stop_condition(self) -> Tuple[bool, int]:
        to_stop = False

        missile_stop, _ = self.missile.check_stop_condition()
        if self.intercepted():  # probable interception
            to_stop = True
            self.flag = 1

        if missile_stop:  # stop due to the missile
            to_stop = True
            self.flag = 2

        if self.is_out_of_view():  # out of field-of-view limit
            to_stop = True
            self.flag = 3

        return to_stop, self.flag

    def intercepted(self) -> bool:
        return self.close_dist_cond.check()

    def is_out_of_view(self) -> bool:
        sigma = self.rel_kin.sigma
        return self.missile.is_out_of_view(sigma)

    def miss_distance(self) -> float:
        x_M = self.missile.history('x')
        x_T = self.target.history('x')

        p_M = x_M[:, 0:2]
        p_T = x_T[:, 0:2]
        d_miss = miss_distance(p_M, p_T)
        return d_miss

    def state_on_closest_instant(self):
        x_M = self.missile.history('x')
        x_T = self.target.history('x')

        p_M = x_M[:, 0:2]
        p_T = x_T[:, 0:2]
        index_close, xi_close = closest_instant(p_M, p_T)
        x_M_close = lin_interp(x_M[index_close], x_M[index_close + 1], xi_close)
        x_T_close = lin_interp(x_T[index_close], x_T[index_close + 1], xi_close)
        return x_M_close, x_T_close

    def report(self):
        self.missile.report()
        if self.flag == 1:
            print("[engagement] The target has been intercepted!")
        else:
            print("[engagement] The target has been missed!")

        d_miss = self.miss_distance()
        x_M_close, x_T_close = self.state_on_closest_instant()

        print("[engagement] Miss distance: {:.6f} (m)".format(d_miss))
        print("[engagement] Missile state on the closest instant: {:.2f}(m), {:.2f}(m), {:.2f}(m/s), {:.2f}(deg) \n".
              format(x_M_close[0], x_M_close[1], x_M_close[2], np.rad2deg(x_M_close[3]))
              )
        print("[engagement] Target state on the closest instant: {:.2f}(m), {:.2f}(m), {:.2f}(m/s), {:.2f}(deg) \n".
              format(x_T_close[0], x_T_close[1], x_T_close[2], np.rad2deg(x_T_close[3]))
              )

    def plot_path(self):
        fig_axs = dict()

        fig_ax = self.missile.plot_path()
        self.target.plot_path(fig_ax)

        fig1, ax1 = fig_ax['fig'], fig_ax['ax']
        fig1.suptitle("2-dim flight path")
        fig_axs['path'] = {'fig': fig1, 'ax': ax1}

        return fig_axs

    def plot_rel_kin(self):
        fig_axs = dict()

        # plots for rel. kin.
        time_list = self.missile.history('t')
        x_M_list = self.missile.history('x')
        x_T_list = self.target.history('x')

        relKin = RelKin2dim()
        r_list = []
        sigma_list = []
        lam_list = []
        omega_list = []

        zem_list = []

        for i in range(x_M_list.shape[0]):
            x_M = x_M_list[i, :]
            x_T = x_T_list[i, :]

            relKin.evaluate(x_M, x_T)

            r_list.append(relKin.r)
            sigma_list.append(relKin.sigma)
            lam_list.append(relKin.lam)
            omega_list.append(relKin.omega)

            zem_list.append(relKin.zem)

        r_list = np.array(r_list)
        sigma_list = np.array(sigma_list)
        lam_list = np.array(lam_list)
        omega_list = np.array(omega_list)

        zem_list = np.array(zem_list)

        fig2, ax2 = plt.subplots(2, 2)
        fig2.tight_layout()
        ax2_1 = ax2[0, 0]
        ax2_1.set_title("Rel. dist")
        ax2_1.plot(time_list[:-1], r_list[:-1], label="Rel. dist")
        ax2_1.set_xlabel("Time (s)")
        ax2_1.set_ylabel("Rel. dist (m)")
        ax2_1.grid()
        ax2_1.legend()

        ax2_2 = ax2[0, 1]
        ax2_2.set_title("Look angle")
        ax2_2.plot(time_list[:-1], np.rad2deg(sigma_list[:-1]), label="look angle")
        ax2_2.set_xlabel("Time (s)")
        ax2_2.set_ylabel("Look angle (deg)")
        ax2_2.grid()
        ax2_2.legend()

        ax2_3 = ax2[1, 0]
        ax2_3.set_title("LOS angle")
        ax2_3.plot(time_list[:-1], np.rad2deg(lam_list[:-1]), label="LOS angle")
        ax2_3.set_xlabel("Time (s)")
        ax2_3.set_ylabel("LOS angle (deg)")
        ax2_3.grid()
        ax2_3.legend()

        ax2_4 = ax2[1, 1]
        ax2_4.set_title("LOS rate")
        ax2_4.plot(time_list[:-1], np.rad2deg(omega_list[:-1]), label="LOS rate")
        ax2_4.set_xlabel("Time (s)")
        ax2_4.set_ylabel("LOS rate (deg/s)")
        ax2_4.grid()
        ax2_4.legend()

        fig_axs['Rel. Kin.'] = {'fig': fig2, 'ax': ax2}

        fig3, ax3 = plt.subplots()
        ax3.set_title("ZEM")
        ax3.plot(time_list[:-1], zem_list[:-1], label="ZEM")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("ZEM (m)")
        ax3.grid()
        ax3.legend()

        fig_axs['Rel. Kin. add.'] = {'fig': fig3, 'ax': ax3}

        return fig_axs


class PurePNG2dimEngagement(Engagement2dim):
    def __init__(self, missile: PlanarMissile2dof, target: PlanarManVehicle2dof):
        super(PurePNG2dimEngagement, self).__init__(missile, target)

        self.pure_png = PurePNG2dim(N=3.0)
        self.attach_sim_objects([self.pure_png])

    # implement
    def forward(self):
        super(PurePNG2dimEngagement, self).forward()
        V_M = self.missile.V
        omega = self.rel_kin.omega

        a_y_cmd = self.pure_png.forward(V_M, omega)
        self.missile.forward(a_M_cmd=np.array([0, a_y_cmd]))


class IACBPNGEngagement(Engagement2dim):
    def __init__(self, missile: PlanarMissile2dof, target: PlanarManVehicle2dof, bpng: IACBPNG):
        super(IACBPNGEngagement, self).__init__(missile, target)
        self.bpng = bpng
        self.attach_sim_objects([self.bpng])

    def forward(self):
        super(IACBPNGEngagement, self).forward()
        a_y_cmd = self.bpng.forward(self.missile, self.target, self.rel_kin)
        self.missile.forward(a_M_cmd=np.array([0., a_y_cmd]))

