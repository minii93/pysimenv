import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable, Optional, Dict, List
from pysimenv.core.util import SimClock, Timer, Logger
from pysimenv.core.error import NoSimClockError


ArrayType = Union[list, tuple, np.ndarray]


class StateVariable(object):
    def __init__(self, state: ArrayType):
        self.state = np.array(state)
        self.deriv = np.zeros_like(state)
        self._rk4_buffer = []
        self.correction_fun = None

    def apply_state(self, state: ArrayType):
        if isinstance(state, float):
            state = np.array([state])
        self.state = np.array(state)

    @property
    def size(self):
        return self.state.size

    def set_deriv(self, deriv: np.ndarray):
        self.deriv = deriv

    def rk4_update_1(self, dt: float):
        x_0 = np.copy(self.state)
        k_1 = np.copy(self.deriv)

        self._rk4_buffer.clear()
        self._rk4_buffer.append(x_0)
        self._rk4_buffer.append(k_1)
        self.state = x_0 + dt / 2 * k_1  # x = x0 + dt/2*k1
        if self.correction_fun is not None:
            self.state = self.correction_fun(self.state)

    def rk4_update_2(self, dt: float):
        x_0 = self._rk4_buffer[0]
        k_2 = np.copy(self.deriv)

        self._rk4_buffer.append(k_2)
        self.state = x_0 + dt / 2 * k_2  # x = x0 + dt/2*k2
        if self.correction_fun is not None:
            self.state = self.correction_fun(self.state)

    def rk4_update_3(self, dt: float):
        x_0 = self._rk4_buffer[0]
        k_3 = np.copy(self.deriv)

        self._rk4_buffer.append(k_3)
        self.state = x_0 + dt / 2 * k_3  # x = x0 + dt/2*k3
        if self.correction_fun is not None:
            self.state = self.correction_fun(self.state)

    def rk4_update_4(self, dt: float):
        x_0, k_1, k_2, k_3 = self._rk4_buffer[:]
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

            state_var.set_deriv(deriv_fun(state_var.state))
            state_var.rk4_update_1(dt)
            state_var.set_deriv(deriv_fun(state_var.state))
            state_var.rk4_update_2(dt)
            state_var.set_deriv(deriv_fun(state_var.state))
            state_var.rk4_update_3(dt)
            state_var.set_deriv(deriv_fun(state_var.state))
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

    def __init__(self, interval: Union[int, float] = -1, name: str = 'model'):
        self.name = name
        self.flag: int = SimObject.FLAG_OPERATING
        self.sim_objs: List[SimObject] = []
        self.state_vars: Dict[str, StateVariable] = dict()
        self._sim_clock: Optional[SimClock] = None
        self._timer = Timer(event_time_interval=interval)
        self._logger = Logger()
        self._last_output = None

    def attach_sim_objs(self, objs: Union['SimObject', list, tuple]):
        if isinstance(objs, SimObject):
            objs = [SimObject]

        for obj in objs:
            if not isinstance(obj, SimObject):
                continue
            self.sim_objs.append(obj)

    def collect_state_vars(self) -> List[StateVariable]:
        state_vars = []
        for var in self.state_vars.values():
            state_vars.append(var)
        for sim_obj in self.sim_objs:
            state_vars.extend(sim_obj.collect_state_vars())
        return state_vars

    @property
    def num_state_vars(self):
        return len(self.state_vars)

    @property
    def num_sim_objs(self):
        return len(self.sim_objs)

    def _attach_sim_clock(self, sim_clock: SimClock):
        self._sim_clock = sim_clock
        self._timer.attach_sim_clock(sim_clock)
        self._timer.turn_on()

    def _attach_log_timer(self, log_timer: Timer):
        self._logger.attach_log_timer(log_timer)

    def _initialize(self):
        pass

    def _detach_sim_clock(self):
        self._sim_clock = None
        self._timer.turn_off()
        self._timer.detach_sim_clock()

    def _detach_log_timer(self):
        self._logger.detach_log_timer()

    def _reset(self):
        self.flag = SimObject.FLAG_OPERATING
        self._timer.reset()
        self._logger.clear()

    def attach_sim_clock(self, sim_clock: SimClock):
        self._attach_sim_clock(sim_clock)
        for sim_obj in self.sim_objs:
            sim_obj.attach_sim_clock(sim_clock)

    def attach_log_timer(self, log_timer: Timer):
        self._attach_log_timer(log_timer)
        for sim_obj in self.sim_objs:
            sim_obj.attach_log_timer(log_timer)

    def initialize(self):
        self._initialize()
        for sim_obj in self.sim_objs:
            sim_obj.initialize()

    def detach_sim_clock(self):
        self._detach_sim_clock()
        for sim_obj in self.sim_objs:
            sim_obj.detach_sim_clock()

    def detach_log_timer(self):
        self._detach_log_timer()
        for sim_obj in self.sim_objs:
            sim_obj.detach_log_timer()
    
    def reset(self):
        self._reset()
        for sim_obj in self.sim_objs:
            sim_obj.reset()

    def check_sim_clock(self):
        assert self._sim_clock is not None, "Attach a sim_clock first!"
        for sim_obj in self.sim_objs:
            sim_obj.check_sim_clock()

    @property
    def time(self) -> float:
        if self._sim_clock is None:
            raise NoSimClockError
        return self._sim_clock.time

    def forward(self, *args, **kwargs):
        self._timer.forward()
        output = self._forward(*args, **kwargs)

        if self._timer.is_event:
            self._last_output = output

        return self._last_output

    # should be implemented
    def _forward(self, *args, **kwargs):
        return NotImplementedError

    @property
    def output(self) -> Union[None, np.ndarray, dict]:
        return self._last_output

    def history(self, *args):
        """
        :param args: variable names
        :return:
        """
        return self._logger.get(*args)

    # to be implemented
    def _check_stop_condition(self) -> Optional[bool]:
        pass

    def check_stop_condition(self) -> bool:
        to_stop_list = [self._check_stop_condition()]
        for sim_obj in self.sim_objs:
            to_stop_list.append(sim_obj.check_stop_condition())

        to_stop = np.array(to_stop_list, dtype=bool).any()
        return to_stop

    def save(self, h5file=None, data_group=''):
        data_group = data_group + '/' + self.name
        self._logger.save(h5file, data_group)
        for sim_obj in self.sim_objs:
            sim_obj.save(h5file, data_group)

    def load(self, h5file=None, data_group=''):
        data_group = data_group + '/' + self.name
        self._logger.load(h5file, data_group)
        for sim_obj in self.sim_objs:
            sim_obj.load(h5file, data_group)


class StaticObject(SimObject):
    def __init__(self, interval: Union[int, float] = -1, eval_fun=None, name='static_obj'):
        super(StaticObject, self).__init__(interval, name)
        self.eval_fun = eval_fun

    # override
    def forward(self, *args, **kwargs) -> Union[None, np.ndarray, dict]:
        self._timer.forward()
        if self._timer.is_event:
            self._last_output = self._forward(*args, **kwargs)

        return self._last_output

    # may be implemented
    def _forward(self, *args, **kwargs) -> Union[None, np.ndarray, dict]:
        if self.eval_fun is None:
            raise NotImplementedError
        else:
            return self.eval_fun(*args, **kwargs)


class BaseFunction(object):
    # to be implemented
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    StateVariable.test()
