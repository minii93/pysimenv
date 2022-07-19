import numpy as np
from typing import Union
from pysimenv.core.base import BaseObject


class MinJerkSol(BaseObject):
    def __init__(self, t_0: float, t_f: float, s_0: np.ndarray, s_f: np.ndarray,
                 interval: Union[int, float] = -1):
        super(MinJerkSol, self).__init__(interval=interval)
        self.t_0 = t_0
        self.t_f = t_f
        self.s_0 = s_0
        self.s_f = s_f

        p_0, v_0, a_0 = s_0[:]
        c = MinJerkSol.calculate_coefficients(t_0, t_f, s_0, s_f)
        alpha, beta, gamma = c[:]
        self.poly = [
            np.array([alpha/120., beta/24., gamma/6., a_0/2., v_0, p_0]),
            np.array([alpha/24., beta/6., gamma/2., a_0, v_0]),
            np.array([alpha/6., beta/2., gamma, a_0])
        ]

    def evaluate(self):
        t = self.time
        s = np.array([
            np.polyval(self.poly[0], t - self.t_0),
            np.polyval(self.poly[1], t - self.t_0),
            np.polyval(self.poly[2], t - self.t_0)])
        return s

    @classmethod
    def calculate_coefficients(cls, t_0: float, t_f: float, s_0: np.ndarray, s_f: np.ndarray):
        """
        :param t_0: initial time
        :param t_f: final time
        :param s_0: initial state (p_0, v_0, a_0)
        :param s_f: final state (p_f, v_f, a_f)
        :return: MinJerkMP object
        np.nan is used for representing an unspecified final state
        """
        if t_0 > t_f:
            raise ValueError

        T = t_f - t_0
        p_0, v_0, a_0 = s_0[:]
        p_f, v_f, a_f = s_f[:]  # any of p_f, v_f, a_f can be np.nan

        delta_p = p_f - p_0 - v_0*T - 1./2*a_0*(T**2)
        delta_v = v_f - v_0 - a_0*T
        delta_a = a_f - a_0

        if np.isfinite(p_f):
            if np.isfinite(v_f):
                if np.isfinite(a_f):
                    # Full defined end state
                    delta = np.array([delta_p, delta_v, delta_a])
                    M = np.array([
                        [720., -360.*T, 60.*(T**2)],
                        [-360.*T, 168.*(T**2), -24.*(T**3)],
                        [60.*(T**2), -24.*(T**3), 3.*(T**4)]
                    ])/T**5
                else:
                    # Fixed position and velocity
                    delta = np.array([delta_p, delta_v])
                    M = np.array([
                        [320., -120.*T],
                        [-200.*T, 72.*(T**2)],
                        [40.*(T**2), -12.*(T**3)]
                    ])/T**5
            else:
                if np.isfinite(a_f):
                    # Fixed position and acceleration
                    delta = np.array([delta_p, delta_a])
                    M = np.array([
                        [90., -15.*(T**2)],
                        [-90.*T, 15.*(T**3)],
                        [30.*(T**2), -3.*(T**4)]
                    ])/T**5/2.
                else:
                    # Fixed position
                    delta = np.array([delta_p])
                    M = np.array([
                        [20.],
                        [-20.*T],
                        [10.*(T**2)]
                    ])/T**5
        else:
            if np.isfinite(v_f):
                if np.isfinite(a_f):
                    # Fixed velocity and acceleration
                    delta = np.array([delta_v, delta_a])
                    M = np.array([
                        [0., 0.],
                        [-12, 6.*T],
                        [6.*T, -2.*(T**2)]
                    ])/T**3
                else:
                    # Fixed velocity
                    delta = np.array([delta_v])
                    M = np.array([
                        [0.],
                        [-3.],
                        [3.*T]
                    ])/T**3
            else:
                if np.isfinite(a_f):
                    # Fixed acceleration
                    delta = np.array([delta_a])
                    M = np.array([
                        [0.],
                        [0.],
                        [1.]
                    ])/T
                else:
                    delta = np.array([-a_0])
                    M = np.array([
                        [0.],
                        [0.],
                        [1.]
                    ])/T
        c = M.dot(delta)  # c = (alpha, beta, gamma)
        return c
