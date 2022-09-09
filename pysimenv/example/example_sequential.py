import numpy as np
from pysimenv.common.model import Sequential, SignalGenerator
from pysimenv.core.base import DynSystem
from pysimenv.core.simulator import Simulator


def main():
    signal_generator = SignalGenerator(lambda t: np.array([np.sin(0.2*np.pi*t)]))

    zeta = 0.5
    omega = 1.
    A = np.array([
        [0., 1.],
        [-omega**2, -2*zeta*omega]
    ])
    B = np.array([[0.], [omega**2]])
    linear_system = DynSystem(
        initial_states={'x': [0., 0.]},
        deriv_fun=lambda x, u: {'x': A.dot(x) + B.dot(u)})

    model = Sequential([
        signal_generator,
        linear_system
    ])
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=10.)
    linear_system.default_plot(show=True)


if __name__ == '__main__':
    main()
