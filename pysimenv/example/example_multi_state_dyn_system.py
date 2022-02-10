import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.system import MultiStateDynSystem
from pysimenv.core.simulator import Simulator


def main():
    def deriv_fun(pos_, vel_, accel_):
        return vel_, accel_

    print("== Test for MultiStateDynSystem ==")
    initial_states = ([0., 0.], [1., 0.])
    accel = np.array([0., 1.])

    model = MultiStateDynSystem(initial_states, deriv_fun)
    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True, accel)

    time = model.history('t')
    pos = model.history('x_0')
    vel = model.history('x_1')

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
