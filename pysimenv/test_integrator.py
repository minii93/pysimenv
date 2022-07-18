from pysimenv.common.model import SignalGenerator, Integrator, Sequential
from pysimenv.core.simulator import Simulator


def main():
    signal_generator = SignalGenerator(
        shaping_fun=lambda t: 1./(1. + t)**2
    )
    integrator = Integrator([0.])
    model = Sequential([signal_generator, integrator])
    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True)
    integrator.default_plot(show=True)


if __name__ == "__main__":
    main()



