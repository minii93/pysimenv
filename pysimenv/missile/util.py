import numpy as np
from typing import Tuple
from pysimenv.core.base import BaseFunction
from pysimenv.common.util import distance_traj_segment


class RelKin2dim(BaseFunction):
    def __init__(self):
        super(RelKin2dim, self).__init__()
        self.p_rel = None
        self.v_rel = None
        self.r = None
        self.los_vec = None
        self.nor_vec = None
        self.lam = None
        self.sigma = None
        self.omega = None

    # implement
    def evaluate(self, x_M: np.ndarray, x_T: np.ndarray):
        """
        :param x_M: (4, ) array, state of the missile
        :param x_T: (4, ) array, state of the target
        :return: None
        """
        p_M, V_M, gamma_M = x_M[0:2], x_M[2], x_M[3]
        p_T, V_T, gamma_T = x_T[0:2], x_T[2], x_T[3]

        v_M = V_M*np.array([np.cos(gamma_M), np.sin(gamma_M)])
        v_T = V_T*np.array([np.cos(gamma_T), np.sin(gamma_T)])

        self.p_rel = p_T - p_M  # relative position
        self.v_rel = v_T - v_M  # relative velocity
        self.r = np.linalg.norm(self.p_rel)  # relative distance
        self.los_vec = self.p_rel / self.r  # LOS unit vector
        self.nor_vec = np.array([-self.los_vec[1], self.los_vec[0]])  # LOS normal vector
        self.lam = np.arctan2(self.los_vec[1], self.los_vec[0])  # LOS angle
        self.sigma = gamma_M - self.lam  # look angle

        los_vec_3 = np.hstack((self.los_vec, 0))
        v_r_3 = np.hstack((self.v_rel, 0))
        omega_3 = np.cross(los_vec_3, v_r_3)/self.r
        self.omega = omega_3[2]

    @property
    def zem(self) -> float:
        z = self.r*self.v_lam / np.linalg.norm(self.v_rel)  # ZEM(Zero-effort miss)
        return z

    @property
    def v_r(self):
        # rel. vel. component parallel to LOS
        return np.dot(self.v_rel, self.los_vec)

    @property
    def v_lam(self):
        # rel. vel. component normal to LOS
        return np.dot(self.v_rel, self.nor_vec)


class CloseDistCond(BaseFunction):
    """
    condition for decreasing distance
    """
    def __init__(self, r_threshold: float = 10.0):
        super(CloseDistCond, self).__init__()
        self.r_threshold = r_threshold  # threshold for the close distance
        self.prev_r = np.inf  # previous relative distance
        self.to_stop = False  # status

    def reset(self):
        self.prev_r = np.inf
        self.to_stop = False

    # implement
    def evaluate(self, r):
        if r < self.r_threshold:
            if r > self.prev_r + 0.01:
                self.to_stop = True
        self.prev_r = r

    def check(self) -> bool:
        return self.to_stop


def miss_distance(p_M: np.ndarray, p_T: np.ndarray, search_range: int = 1) -> float:
    num_sample = p_M.shape[0]
    d = np.linalg.norm(p_T - p_M, axis=1)
    index_c = np.argmin(d)
    d_miss = d[index_c]

    index_min = max(index_c - search_range, 0)
    index_max = min(index_c + search_range, num_sample - 1)
    for i in range(index_min, index_max):
        p0 = p_M[i, :]
        p1 = p_M[i + 1, :]
        q0 = p_T[i, :]
        q1 = p_T[i + 1, :]

        d_, _ = distance_traj_segment(p0, p1, q0, q1)
        d_miss = min(d_miss, d_)
    return d_miss


def closest_instant(p_M: np.ndarray, p_T: np.ndarray, search_range: int = 1) -> Tuple[int, float]:
    num_sample = p_M.shape[0]
    d = np.linalg.norm(p_T - p_M, axis=1)
    index_c = np.argmin(d)
    d_miss = d[index_c]

    index_min = max(index_c - search_range, 0)
    index_max = min(index_c + search_range, num_sample - 1)

    index_close = index_min
    xi_close = 0.
    for i in range(index_min, index_max):
        p0 = p_M[i, :]
        p1 = p_M[i + 1, :]
        q0 = p_T[i, :]
        q1 = p_T[i + 1, :]

        d_, xi_ = distance_traj_segment(p0, p1, q0, q1)
        if d_ < d_miss:
            index_close = i
            xi_close = xi_
            d_miss = d_

    return index_close, xi_close


def lin_interp(x_i: np.ndarray, x_f: np.ndarray, xi: float) -> np.ndarray:
    x = x_i + xi*(x_f - x_i)
    return x
