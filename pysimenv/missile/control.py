import numpy as np
import scipy
import matplotlib.pyplot as plt
from pysimenv.core.base import SimObject


class PitchAP(SimObject):
    def __init__(self, V, L_alp, L_delta, M_alp, M_q, M_delta, tau, *args, **kwargs):
        super(PitchAP, self).__init__()
        A = np.array([
            [-L_alp/V, L_alp, -L_delta/tau],
            [M_alp/L_alp, M_q, M_delta - M_alp*L_delta/L_alp],
            [0., 0., -1./tau]
        ])
        B = np.array([[L_delta/tau], [0.], [1./tau]])

        C = np.array([[1., 0., 0.]])

        Q = np.diag([1., 0.01, 0.])
        R = np.diag([10000.])
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R).dot(B.transpose().dot(P))

        A_c = A - B.dot(self.K)
        e_1 = np.array([1., 0., 0.])
        self.K_ss = -1./(C.dot(np.linalg.inv(A_c).dot(B).dot(self.K).dot(e_1)))[0]

    # implement
    def _forward(self, a_L, q, delta, a_L_c) -> float:
        e = np.array([a_L - self.K_ss*a_L_c, q, delta])
        delta_c = -self.K.dot(e)[0]

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
