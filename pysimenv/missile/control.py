import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.base import SimObject


class ThreeLoopAP(SimObject):
    def __init__(self, V, L_alp, M_q, M_delta, tau_d, omega_d, zeta_d, name="autopilot", **kwargs):
        super(ThreeLoopAP, self).__init__(name=name, **kwargs)
        self.K_A = 1./(L_alp*tau_d) - 1./V
        self.K_DC = V/(V - L_alp*tau_d)
        self.K_R = (2*zeta_d*omega_d + M_q)/M_delta
        self.omega_i = omega_d**2/(2*zeta_d*omega_d + M_q)
        self._add_state_vars(e_q_i=np.array([0.]))

    def _forward(self, q, a_L, a_L_c) -> float:
        q_c = self.K_A*(self.K_DC*a_L_c - a_L)
        e_q_i = self.state('e_q_i')[0]
        delta_c = self.K_R*(self.omega_i*e_q_i - q)

        e_q = q_c - q
        self.state_vars['e_q_i'].set_deriv(deriv=e_q)

        self._logger.append(t=self.time, a_L=a_L, a_L_c=a_L_c)
        return delta_c

    def plot_accel(self, show=False):
        data = self.history('t', 'a_L', 'a_L_c')

        fig, ax = plt.subplots()
        ax.plot(data['t'], data['a_L'], label="Actual")
        ax.plot(data['t'], data['a_L_c'], label="Command")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s^2)")
        ax.grid()
        ax.legend()

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)
