from pysimenv.core.system import TimeVaryingDynSystem
from pysimenv.core.simulator import Simulator


def main():
    print("== Test for TimeVaryingDynSystem ==")
    dt = 0.01

    def deriv_fun(t, x):
        return {'x': -(1./(1. + t)**2)*x}

    model = TimeVaryingDynSystem(
        initial_states={'x': [1.]},
        deriv_fun=deriv_fun
    )
    simulator = Simulator(model)
    simulator.propagate(dt=dt, time=10., save_history=True)
    model.default_plot(show=True)


if __name__ == "__main__":
    main()
