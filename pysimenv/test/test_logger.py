import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.util import SimClock, Logger


def main():
    dt = 0.01
    sim_clock = SimClock()
    logger = Logger()

    A = np.array(
        [[1., dt], [0., 1.]]
    )
    B = np.array([[0.], [dt]])
    x = np.array([0., 0.])
    u = np.array([1.])

    for i in range(100):
        logger.append(time=sim_clock.time, state=x, control=u)
        x = A.dot(x) + B.dot(u)
        sim_clock.elapse(dt)

    logged_data = logger.get()
    fig, ax = plt.subplots()
    ax.plot(logged_data['time'], logged_data['state'], label={"Pos. [m]", "Vel. [m/s]"})
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State")
    ax.grid()
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
