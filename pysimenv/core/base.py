import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable, Optional, Tuple
from pysimenv.core.util import SimClock, Timer, Logger
from pysimenv.core.error import NoSimClockError


ArrayType = Union[list, tuple, np.ndarray]


class StateVariable(object):
    def __init__(self, state: ArrayType):
        self.state = np.array(state)
        self.deriv = np.zeros_like(state)
        self.rk4_buffer = []
        self.correction_fun = None

    def apply_state(self, state: ArrayType):
        self.state = np.array(state)

    @property
    def size(self):
        return self.state.size

    def forward(self, deriv: np.ndarray):
        self.deriv = deriv

    def rk4_update_1(self, dt: float):
        x_0 = np.copy(self.state)
        k_1 = np.copy(self.deriv)

        self.rk4_buffer.clear()
        self.rk4_buffer.append(x_0)
        self.rk4_buffer.append(k_1)
        self.state = x_0 + dt / 2 * k_1  # x = x0 + dt/2*k1
        if self.correction_fun is not None:
            self.state = self.correction_fun(self.state)

    def rk4_update_2(self, dt: float):
        x_0 = self.rk4_buffer[0]
        k_2 = np.copy(self.deriv)

        self.rk4_buffer.append(k_2)
        self.state = x_0 + dt / 2 * k_2  # x = x0 + dt/2*k2
        if self.correction_fun is not None:
            self.state = self.correction_fun(self.state)

    def rk4_update_3(self, dt: float):
        x_0 = self.rk4_buffer[0]
        k_3 = np.copy(self.deriv)

        self.rk4_buffer.append(k_3)
        self.state = x_0 + dt / 2 * k_3  # x = x0 + dt/2*k3
        if self.correction_fun is not None:
            self.state = self.correction_fun(self.state)

    def rk4_update_4(self, dt: float):
        x_0, k_1, k_2, k_3 = self.rk4_buffer[:]
        k_4 = np.copy(self.deriv)

        self.state = x_0 + dt * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
        if self.correction_fun is not None:
            self.state = self.correction_fun(self.state)

    def attach_correction_fun(self, correction_fun: Callable[[np.ndarray], np.ndarray]):
        self.correction_fun = correction_fun

    @staticmethod
    def test():
        state_var = StateVariable([1.])
        def deriv_fun(x): return -x
        dt = 0.01

        time_list = []
        sim_state_list = []
        t = 0
        for i in range(1000):
            time_list.append(t)
            sim_state_list.append(state_var.state)

            state_var.forward(deriv_fun(state_var.state))
            state_var.rk4_update_1(dt)
            state_var.forward(deriv_fun(state_var.state))
            state_var.rk4_update_2(dt)
            state_var.forward(deriv_fun(state_var.state))
            state_var.rk4_update_3(dt)
            state_var.forward(deriv_fun(state_var.state))
            state_var.rk4_update_4(dt)
            t = t + dt

        time_list = np.array(time_list)
        sim_state_list = np.array(sim_state_list)
        true_state_list = np.exp(-time_list)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(time_list, sim_state_list[:, 0], color='k')
        plt.xlabel("Time [s]")
        plt.ylabel("State")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(time_list, sim_state_list[:, 0] - true_state_list, color='k')
        plt.xlabel("Time [s]")
        plt.ylabel("Integration error")
        plt.grid()

        plt.subplots_adjust(bottom=0.1, left=0.1, top=0.9, hspace=0.5)
        plt.show()


class SimObject(object):
    FLAG_OPERATING = 0

    def __init__(self, interval: Union[int, float] = -1):
        self.flag: int = SimObject.FLAG_OPERATING
        self.sim_clock: Optional[SimClock] = None
        self.log_timer: Optional[Timer] = None
        self.timer = Timer(event_time_interval=interval)
        self.logger = Logger()

    def attach_sim_clock(self, sim_clock: SimClock):
        self.sim_clock = sim_clock
        self.timer.attach_sim_clock(sim_clock)
        self.timer.turn_on()

    def attach_log_timer(self, log_timer: Timer):
        self.log_timer = log_timer

    def initialize(self):
        pass

    def detach_sim_clock(self):
        self.sim_clock = None
        self.timer.turn_off()
        self.timer.detach_sim_clock()

    def detach_log_timer(self):
        self.log_timer = None
    
    def reset(self):
        self.flag = SimObject.FLAG_OPERATING
        self.timer.reset()
        self.logger.clear()

    @property
    def time(self) -> float:
        if self.sim_clock is None:
            raise NoSimClockError
        return self.sim_clock.time

    # to be implemented
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def output(self) -> Union[None, tuple, np.ndarray]:
        return self._output()

    # to be implemented
    def _output(self) -> Union[None, tuple, np.ndarray]:
        return None

    # to be implemented
    def check_stop_condition(self) -> Tuple[bool, int]:
        to_stop = False
        flag = self.flag
        return to_stop, flag


class StaticObject(SimObject):
    def __init__(self, interval: Union[int, float] = -1, eval_fun=None):
        super(StaticObject, self).__init__(interval)
        self.eval_fun = eval_fun
        self.last_output = None

    def forward(self, *args, **kwargs):
        self.timer.forward()
        if self.timer.is_event:
            if self.eval_fun is None:
                self.last_output = self.evaluate(*args, **kwargs)
            else:
                self.last_output = self.eval_fun(*args, **kwargs)

        return self.last_output

    def _output(self) -> Union[None, list, np.ndarray]:
        return self.last_output

    # to be implemented
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError


class BaseFunction(object):
    # to be implemented
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    StateVariable.test()
