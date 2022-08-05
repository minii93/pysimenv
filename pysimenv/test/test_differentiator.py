import numpy as np
import matplotlib.pyplot as plt
from pysimenv.common.model import Differentiator, SignalGenerator, Scope, Sequential
from pysimenv.core.simulator import Simulator


def main():
    signal_generator = SignalGenerator(lambda t: np.array([1. - np.exp(-0.2*t)]))
    derivative = Differentiator()
    scope = Scope()

    model = Sequential([
        signal_generator,
        derivative,
        scope
    ])
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=10.)

    t = scope.history('t')
    y_numerical = scope.history('u')
    y_analytic = 0.2*np.exp(-0.2*t)

    fig, ax = plt.subplots()
    ax.plot(t, y_numerical, label="Numerical")
    ax.plot(t, y_analytic, label="Analytic")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Derivative")
    ax.grid()
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()

