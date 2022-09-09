import numpy as np
from scipy.optimize import fmin_slsqp
from pysimenv.multicopter.model import QuadXThrustModel
from pysimenv.multicopter.utils import RelaxedHoverSolver

m = 0.5
J = np.diag([2.7, 2.7, 5.2])*0.001
g = 9.807
d_phi = 0.17/np.sqrt(2.)
d_theta = 0.17/np.sqrt(2.)
kappa_f = 6.41e-6
kappa_tau = 1.1e-7
G = np.array([
    [kappa_f, kappa_f, kappa_f, kappa_f],
    [-kappa_f*d_phi, kappa_f*d_phi, kappa_f*d_phi, -kappa_f*d_phi],
    [kappa_f*d_theta, -kappa_f*d_theta, kappa_f*d_theta, -kappa_f*d_theta],
    [kappa_tau, kappa_tau, -kappa_tau, -kappa_tau]
])


def eq_const(x: np.ndarray):
    omega = x[0:3]
    v_op = x[3:5]

    v = np.array([v_op[0], v_op[1], 0., 0.])

    u = G.dot(np.square(v))
    p, q, r = omega
    J_x, J_y, J_z = np.diag(J)[:]

    c_1 = np.array([m*g - u[0]])
    c_2 = np.array([
        (J_z - J_y)*q*r - u[1],
        (J_x - J_z)*p*r - u[2],
        (J_y - J_x)*p*q - u[3]
    ])
    return np.hstack((c_1, c_2))


def ieq_const(x: np.ndarray):
    v_op = x[3:5]
    return v_op


def obj_fun(x: np.ndarray):
    omega = x[0:3]
    return np.sum(np.square(omega))


def main():
    omega_0 = np.zeros(3)
    v_op_0 = np.sqrt(np.ones(2)*m*g/2./kappa_f)
    x_0 = np.hstack((omega_0, v_op_0))

    x_opt = fmin_slsqp(obj_fun, x_0, eqcons=[eq_const], ieqcons=[ieq_const])
    print("-------")
    print(x_opt)


if __name__ == "__main__":
    main()
