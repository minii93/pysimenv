import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from pysimenv.core.base import SimObject
from pysimenv.missile.model import PlanarMissile, PlanarVehicle
from pysimenv.missile.util import RelKin2dim


class Guidance2dim(SimObject):
    def __init__(self, interval: Union[int, float] = -1):
        super(Guidance2dim, self).__init__(interval=interval)

    # to be implemented
    def _forward(self, missile: PlanarMissile, target: PlanarVehicle, rel_kin: RelKin2dim) -> float:
        raise NotImplementedError


class PurePNG2dim(Guidance2dim):
    def __init__(self, N: float = 3.0, interval: Union[int, float] = -1):
        super(PurePNG2dim, self).__init__(interval=interval)
        self.N = N

    # implement
    def _forward(self, missile: PlanarMissile, target: PlanarVehicle, rel_kin: RelKin2dim) -> float:
        """
        :return: acceleration command a_y_cmd
        """
        V_M = missile.kin.V  # speed of the missile
        omega = rel_kin.omega  # LOS rate

        a_M_cmd = self.N*V_M*omega
        return a_M_cmd


class IACBPNG(Guidance2dim):
    """
    Biased PNG with terminal-angle constraint (impact angle control)
    """
    def __init__(self, N: float, tau: float, theta_M_0: float, theta_M_f: float, theta_T: float, lam_0: float,
                 sigma_max: float = float('inf'), b_max: float = float('inf')):
        super(IACBPNG, self).__init__()
        self._add_state_vars(B=np.array([0.]))

        # Guidance parameters
        self.N = N
        self.tau = tau
        self.theta_M_0 = theta_M_0
        self.theta_M_f = theta_M_f
        self.theta_T = theta_T
        self.lam_0 = lam_0
        self.sigma_max = sigma_max
        self.b_max = b_max

        self.phase = 0  # 0: BPNG, 1: DPP, 2: BPNG

    @classmethod
    def reference_value(cls, theta_M_0, theta_M_f, theta_T, lam_0, N, V_M, V_T):
        theta_M_0_bar = theta_M_0 - lam_0
        theta_M_f_bar = theta_M_f - lam_0
        theta_T_bar = theta_T - lam_0

        lam_f_bar = np.arctan(
            (V_M*np.sin(theta_M_f_bar) - V_T*np.sin(theta_T_bar))/(V_M*np.cos(theta_M_f_bar) - V_T*np.cos(theta_T_bar))
        )
        B_ref = theta_M_f_bar - theta_M_0_bar - N*lam_f_bar
        return B_ref

    def _forward(self, missile: PlanarMissile, target: PlanarVehicle, rel_kin: RelKin2dim) -> float:
        V_M = missile.V
        V_T = target.V
        omega = rel_kin.omega
        sigma = missile.look_angle(rel_kin.lam)

        B_ref = IACBPNG.reference_value(
            self.theta_M_0, self.theta_M_f, self.theta_T, self.lam_0, self.N, V_M, V_T)
        B = self.state('B')[0]

        b_1 = 1./self.tau*(B_ref - B)
        b_2 = (1. - self.N)*omega
        if self.phase == 0:
            # BPNG
            b = b_1
            if np.abs(sigma) > self.sigma_max - np.deg2rad(0.01):
                self.phase = 1
        elif self.phase == 1:
            # DPP
            b = b_2
            if np.abs(b_1) < np.abs(b_2):
                self.phase = 2
        else:
            # BPNG
            b = b_1

        b = np.clip(b, -self.b_max, self.b_max)
        a_png = self.N*V_M*omega
        a_b = V_M*b
        a_M = a_png + a_b

        self.state_vars['B'].set_deriv(deriv=b)
        self._logger.append(t=self.time, B_ref=B_ref, B=B, b=b, a_png=a_png, a_b=a_b)
        return a_M

    def plot_bias(self):
        t = self.history('t')
        b = self.history('b')

        fig, ax = plt.subplots()
        ax.plot(t, b)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Bias, b (rad/s)")
        ax.grid()

    def plot_bias_integral(self):
        t = self.history('t')
        B_ref = self.history('B_ref')
        B = self.history('B')

        fig, ax = plt.subplots()
        ax.plot(t, B_ref, linestyle='--')
        ax.plot(t, B)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Integral Value of Bias, B (rad)")
        ax.grid()
