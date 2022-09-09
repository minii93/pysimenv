import numpy as np
from pysimenv.core.base import StaticObject, DynSystem
from pysimenv.core.simulator import Simulator
from pysimenv.common.model import OFBControl


def main():
    """
    Reference: B. L. Stevens, F. L. Lewis, E. N. Johnson, "Aircraft Control and Simulation", 3rd edition
    Example 5.3-1: LQR Design for F-16 Lateral Regulator (pp.407)
    State:
        x = [beta, phi, p, r, delta_a, delta_r, r_w]
        beta, phi: [rad], p, r: [rad/s], delta_a, delta_r: [deg], r_w: [deg]
    Input:
        u = [u_a, u_r]
        u_a: aileron servo input [deg]
        u_r: rudder servo input [deg]
    Output:
        y = [r_w, p, beta, phi]
        r_w, beta, phi: [deg], p: [deg/s]
    """
    A = np.array([
        [-0.3220, 0.0640, 0.0364, -0.9917, 0.0003, 0.0008, 0],
        [0, 0, 1, 0.0037, 0, 0, 0],
        [-30.6492, 0, -3.6784, 0.6646, -0.7333, 0.1315, 0],
        [8.5396, 0, -0.0254, -0.4764, -0.0319, -0.0620, 0],
        [0, 0, 0, 0, -20.2, 0, 0],
        [0, 0, 0, 0, 0, -20.2, 0],
        [0, 0, 0, 57.2958, 0, 0, -1]
    ])
    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [20.2, 0],
        [0, 20.2],
        [0, 0]
    ])
    C = np.array([
        [0, 0, 0, 57.2958, 0, 0, -1],
        [0, 0, 57.5928, 0, 0, 0, 0],
        [57.5928, 0, 0, 0, 0, 0, 0],
        [0, 57.5928, 0, 0, 0, 0, 0]
    ])
    K = np.array([
        [-0.56, -0.44, 0.11, -0.35],
        [-1.19, -0.21, -0.44, 0.26]])

    linear_system = DynSystem(
        initial_states={'x': [1., 0., 0., 0., 0., 0., 0.]},
        deriv_fun=lambda x, u: {'x': A.dot(x) + B.dot(u)},
        output_fun=lambda x: C.dot(x)
    )
    control = StaticObject(interval=-1, eval_fun=lambda y: -K.dot(y))
    model = OFBControl(linear_system, control)  # closed-loop system

    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=10.)
    linear_system.default_plot(
        show=True,
        var_ind_dict={'x': [0, 3, 1, 2]},
        var_names_dict={'x': ['beta', 'r', 'phi', 'p'], 'u': ['u_a', 'u_r']}
    )


if __name__ == "__main__":
    main()
