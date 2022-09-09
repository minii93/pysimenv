import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.base import DynSystem
from pysimenv.core.simulator import Simulator


def main():
    def deriv_fun(x, u):
        omega = 1.
        zeta = 0.8
        A = np.array([[0, 1.], [-omega**2, -2*zeta*omega]])
        B = np.array([[0.], [omega**2]])
        x_dot = A.dot(x) + B.dot(u)
        return {'x': x_dot}

    sys = DynSystem(
        initial_states={'x': np.zeros(2)},
        deriv_fun=deriv_fun
    )
    simulator = Simulator(sys)
    simulator.propagate(dt=0.01, time=10., save_history=True, u=np.array([1.]))

    t = sys.history('t')
    x = sys.history('x')
    u = sys.history('u')

    fig, ax = plt.subplots()
    for i in range(2):
        ax.plot(t, x[:, i], label="x_" + str(i + 1))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("x")
    ax.grid()
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(t, u)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("u")
    ax.grid()

    plt.show()


if __name__ == "__main__":
    main()
