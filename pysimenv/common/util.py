import numpy as np
from typing import Union, Tuple
from scipy.optimize import minimize_scalar


ArrayType = Union[list, tuple, np.ndarray]


def wrap_to_pi(angle: Union[float, np.ndarray]):
    if isinstance(angle, np.ndarray):
        angle_w = angle.copy()
        angle_w[angle > np.pi] = angle[angle > np.pi] - 2*np.pi
        angle_w[angle < -np.pi] = angle[angle < -np.pi] + 2*np.pi
    else:
        angle_w = angle
        if angle_w > np.pi:
            angle_w -= 2*np.pi
        elif angle_w < -np.pi:
            angle_w += 2*np.pi
    return angle_w


def distance_traj_segment(p0: ArrayType, p1: ArrayType, q0: ArrayType, q1: ArrayType) -> Tuple[float, float]:
    """
    A function finding the distance between two trajectory segements
    :param p0: the initial point of trajectory segment 1
    :param p1: the final point of trajectory segment 1
    :param q0: the initial point of trajectory segment 2
    :param q1: the final point of trajectory segment 2
    :return: distance
    """
    p0_ = np.array(p0)
    p1_ = np.array(p1)
    q0_ = np.array(q0)
    q1_ = np.array(q1)

    def sq_distance(xi: float):
        p = p0_ + xi * (p1_ - p0_)
        q = q0_ + xi * (q1_ - q0_)
        return np.sum((p - q)**2)
    res = minimize_scalar(sq_distance, bounds=(0, 1), method='bounded')
    d = np.sqrt(res.fun)
    xi_opt = res.x
    return d, xi_opt
