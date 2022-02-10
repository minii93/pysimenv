import numpy as np
import matplotlib.pyplot as plt
from pytictoc import TicToc
from pysimenv.core.util import SimClock, Timer
from pysimenv.core.system import BaseSystem, TimeInvarDynSystem


class Simulator(object):
    def __init__(self, model: BaseSystem, verbose: bool = True):
        self.sim_clock = SimClock()
        self.log_timer = Timer(np.inf)
        self.log_timer.attach_sim_clock(self.sim_clock)

        self.model = model
        self.model.attach_sim_clock(self.sim_clock)
        self.model.attach_log_timer(self.log_timer)
        self.model.initialize()

        self.verbose = verbose

    def reset(self):
        self.sim_clock.reset()
        self.log_timer.turn_off()
        self.model.reset()

    def begin_logging(self, log_interval: float):
        self.log_timer.turn_on(log_interval)

    def finish_logging(self):
        self.log_timer.turn_off()

    def step(self, dt: float, *args, **kwargs):
        self.model.step(dt, *args, **kwargs)

    def propagate(self, dt: float, time: float, save_history: bool = True, *args, **kwargs):
        if save_history:
            self.begin_logging(dt)

        if self.verbose:
            print("[simulator] Simulating...")
        tic_toc = TicToc()
        tic_toc.tic()
        self.model.propagate(dt, time, *args, **kwargs)
        elapsed_time = tic_toc.tocvalue()

        if self.verbose:
            print("[simulator] Elapsed time: {:.4f} [s]".format(elapsed_time))
        self.finish_logging()

    @staticmethod
    def test():
        def deriv_fun(x, u):
            A = np.array([[0, 1], [-1, -1]])
            B = np.array([0, 1])
            return A.dot(x) + B.dot(u)

        model = TimeInvarDynSystem([0., 1.], deriv_fun)
        simulator = Simulator(model)
        simulator.propagate(dt=0.01, time=10., save_history=True, u=1.)
        model.default_plot()

        simulator.reset()
        simulator.begin_logging(0.01)
        for i in range(1000):
            simulator.step(dt=0.01, u=1.)
        simulator.finish_logging()
        model.default_plot()

        plt.show()


if __name__ == "__main__":
    Simulator.test()
