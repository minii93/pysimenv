import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.base import DynSystem
from pysimenv.core.simulator import Simulator


def main():
    def deriv_fun(p, v, a):
        p_dot = v.copy()
        v_dot = a.copy()
        return {'p': p_dot, 'v': v_dot}

    print("== Test for MultiStateDynSystem ==")
    initial_states = {'p': [0., 0.], 'v': [1., 0.]}
    a = np.array([0., 1.])

    model = DynSystem(
        initial_states=initial_states,
        deriv_fun=deriv_fun
    )
    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True, a=a)

    time = model.history('t')
    pos = model.history('p')
    vel = model.history('v')

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, pos, label={"x [m]", "y [m]"})
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, vel, label={"v_x [m/s]", "v_y [m/s]"})
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
