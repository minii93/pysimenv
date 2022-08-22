import numpy as np
import scipy.linalg
from typing import Tuple
from pysimenv.core.base import ArrayType


def check_orthogonality(M: np.ndarray) -> bool:
    diff = np.linalg.norm(np.matmul(M, np.transpose(M)) - np.identity(3))
    is_orthogonal = (diff < 1e-6)
    return is_orthogonal


def correct_orthogonality(M: np.ndarray) -> np.ndarray:
    sq_M = np.matmul(M, np.transpose(M))
    sqrt_M = np.real(scipy.linalg.sqrtm(sq_M))
    return np.linalg.solve(sqrt_M, M)


def check_skew_symmetry(M: np.ndarray) -> bool:
    diff = np.linalg.norm(M + np.transpose(M))
    is_skew_symmetric = (diff < 1e-4)
    return is_skew_symmetric


def correct_skew_symmetry(M: np.ndarray) -> np.ndarray:
    return (M - np.transpose(M))/2


def hat(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.]
    ])


def basic_rotation(axis: str, phi: float) -> np.ndarray:
    """
    Suppose that the body frame {b} is rotated from
    the inertial frame {i} by phi with respect to x-axis.
    Then the rotation matrix R_{bi} from {i} to {b} can be
    expressed as
    R_{bi} = np.array([
        [1, 0, 0],
        [0, cos(phi), sin(phi)],
        [0, -sin(phi), cos(phi)]])
    and p_{b} = R_{bi}p_{i}
    :param axis: the axis of rotation, one of 'x', 'y', 'z'
    :param phi: the angle of rotation, scalar
    :return: rotation matrix, (3, 3) numpy array
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c_phi, s_phi],
            [0, -s_phi, c_phi]
        ], dtype=np.float32)
    elif axis == 'y':
        return np.array([
            [c_phi, 0, -s_phi],
            [0, 1, 0],
            [s_phi, 0, c_phi]
        ], dtype=np.float32)
    else:
        # when axis == 'z'
        return np.array([
            [c_phi, s_phi, 0],
            [-s_phi, c_phi, 0],
            [0, 0, 1]
        ], dtype=np.float32)


def euler_angles_to_rotation(eta: ArrayType) -> np.ndarray:
    """
    convention: 3-2-1 Euler angle
    The resulting rotation matrix is from inertial frame
    to body frame, R_{bi}
    :param eta: eulerAngles = [phi, theta, psi]
    :return: rotation matrix, (3, 3) numpy array
    """
    phi, theta, psi = eta[:]

    s_phi = np.sin(phi)
    c_phi = np.cos(phi)
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    s_psi = np.sin(psi)
    c_psi = np.cos(psi)

    R_psi = np.array([
        [c_psi, s_psi, 0.],
        [-s_psi, c_psi, 0.],
        [0., 0., 1.]
    ])
    R_theta = np.array([
        [c_theta, 0., -s_theta],
        [0., 1., 0.],
        [s_theta, 0., c_theta]
    ])
    R_phi = np.array([
        [1., 0., 0.],
        [0., c_phi, s_phi],
        [0., -s_phi, c_phi]
    ])
    return np.matmul(R_phi, np.matmul(R_theta, R_psi))


def rotation_to_euler_angles(R: np.ndarray) -> list:
    r_11 = R[0, 0]
    r_12 = R[0, 1]
    r_13 = R[0, 2]
    r_23 = R[1, 2]
    r_33 = R[2, 2]

    if abs(r_13) < 1 - 1e-6:
        theta = np.arctan2(-r_13, np.sqrt(r_11**2 + r_12**2))
        psi = np.arctan2(r_12, r_11)
        phi = np.arctan2(r_23, r_33)
    else:
        r_21 = R[1, 0]
        r_22 = R[1, 1]
        if r_13 < 0:
            theta = np.pi/2
            psi = 0
            phi = np.arctan2(r_21, r_22)
        else:
            theta = -np.pi/2
            psi = 0
            phi = -np.arctan2(r_21, r_22)

    return [phi, theta, psi]


def axis_angle_to_quaternion(a: np.ndarray, phi: float) -> np.ndarray:
    q = np.hstack((
        np.array([np.cos(phi/2.)]),
        np.sin(phi/2.)*a
    ))
    return q


def quaternion_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    a = q[1:4]
    if np.linalg.norm(a) > 1e-8:
        a = a/np.linalg.norm(a)
    q_0 = max(-1., min(1., q[0]))
    phi = 2*np.arccos(q_0)
    return a, phi


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    eta = q[0]
    epsilon = q[1:4]
    R = (eta**2 - epsilon.dot(epsilon))*np.identity(3) + 2*np.outer(epsilon, epsilon) - 2*eta*hat(epsilon)
    return R


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    r_11 = R[0, 0]
    r_22 = R[1, 1]
    r_33 = R[2, 2]

    r_12 = R[0, 1]
    r_21 = R[1, 0]
    r_23 = R[1, 2]
    r_32 = R[2, 1]
    r_31 = R[2, 0]
    r_13 = R[0, 2]

    eta = 0.5*np.sqrt(max(0., 1. + np.trace(R)))
    epsilon_1 = np.sign(r_23 - r_32)*0.5*np.sqrt(
        max(0., 1. + r_11 - r_22 - r_33)
    )
    epsilon_2 = np.sign(r_31 - r_13)*0.5*np.sqrt(
        max(0., 1. - r_11 + r_22 - r_33)
    )
    epsilon_3 = np.sign(r_12 - r_21)*0.5*np.sqrt(
        max(0., 1. - r_11 - r_22 + r_33)
    )

    q = np.array([eta, epsilon_1, epsilon_2, epsilon_3])
    return q
