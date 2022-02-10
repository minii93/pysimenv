import matplotlib.pyplot as plt
from pysimenv.common.model import FirstOrderLinSys, SecondOrderLinSys
from pysimenv.core.simulator import Simulator


def main():
    lin_sys_1 = FirstOrderLinSys([0.], tau=0.5)
    simulator = Simulator(lin_sys_1)
    simulator.propagate(dt=0.01, time=10., save_history=True, u=1.)
    lin_sys_1.default_plot()

    lin_sys_2 = SecondOrderLinSys([0., 0.], zeta=0.1, omega=1.)
    simulator = Simulator(lin_sys_2)
    simulator.propagate(dt=0.01, time=10., save_history=True, u=1.)
    lin_sys_2.default_plot()
    plt.show()


if __name__ == '__main__':
    main()

