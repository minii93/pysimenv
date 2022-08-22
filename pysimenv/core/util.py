import numpy as np
from typing import Optional, Union
from pysimenv.core.error import NoSimClockError


class SimClock(object):
    def __init__(self, time: float = 0., time_res: float = 1e-6):
        self.time = time  # Current time
        self.time_res = time_res  # Time resolution
        self.dt = None
        self.major_time_step: bool = True

    def reset(self):
        self.time = 0.
        self.dt = None

    def apply_time(self, time: float):
        self.time = time

    def set_time_res(self, time_res: float):
        self.time_res = time_res

    def set_time_interval(self, dt: float):
        self.dt = dt

    def elapse(self, dt: float):
        self.time += dt


class Timer(object):
    def __init__(self, event_time_interval: Union[int, float] = -1):
        # construction phase
        self.sim_clock: Optional[SimClock] = None
        self.event_time_interval = event_time_interval

        self.is_operating = False
        self.is_event = False
        self._last_event_time = -np.inf

    def reset(self):
        self.turn_off()

    def attach_sim_clock(self, sim_clock: SimClock):
        # initialization phase
        self.sim_clock = sim_clock

    def turn_on(self, event_time_interval: Union[None, int, float] = None):
        # initialization phase
        if self.sim_clock is None:
            raise NoSimClockError()
        if event_time_interval is not None:
            self.event_time_interval = event_time_interval

        self.is_operating = True
        self.is_event = True
        self._last_event_time = self.sim_clock.time

    def turn_off(self):
        self.is_operating = False
        self.is_event = False
        self._last_event_time = -np.inf

    def detach_sim_clock(self):
        self.sim_clock = None

    def forward(self):
        if self.is_operating:
            # prevent repetitive calling
            if abs(self.sim_clock.time - self._last_event_time) <= self.sim_clock.time_res:
                return

            # always raise an event when the event time interval is -1
            if self.event_time_interval == -1:
                self.is_event = True
                self._last_event_time = self.sim_clock.time
                return

            elapsed_time = self.sim_clock.time - self._last_event_time
            if elapsed_time >= (self.event_time_interval - self.sim_clock.time_res):
                self.is_event = True
                self._last_event_time = self.sim_clock.time
            else:
                self.is_event = False

    @staticmethod
    def test():
        print("== Test for Timer ==")

        dt = 0.01
        sim_clock = SimClock()

        event_time_interval = 0.1
        timer = Timer(event_time_interval)
        timer.attach_sim_clock(sim_clock)
        timer.turn_on()

        print("Initial time: 0.0[s]")
        print("Sampling time interval: {:.2f}[s]".format(dt))
        print("Event time interval: {:.2f}[s]\n".format(event_time_interval))
        for i in range(100):
            timer.forward()
            if timer.is_event:
                print("[Timer] An event occurred at time = {:.1f}[s]".format(sim_clock.time))
            sim_clock.elapse(dt)


class Logger:
    def __init__(self):
        self._log_timer: Optional[Timer] = None
        self.data = dict()
        self.is_operating = True

    def attach_log_timer(self, log_timer: Timer):
        self._log_timer = log_timer

    def turn_on(self):
        self.is_operating = True

    def turn_off(self):
        self.is_operating = False

    def detach_log_timer(self):
        self._log_timer = None

    def clear(self):
        self.data = dict()

    def is_empty(self):
        return (self.data is None) or (len(self.data) == 0)

    def len(self):
        if self.data is None:
            return 0
        first_key = next(iter(self.data.keys()))
        return len(self.data[first_key])

    def append(self, **kwargs):
        if self.is_operating:
            if (self._log_timer is None) or self._log_timer.is_event:
                for key in kwargs.keys():
                    arr = kwargs[key]
                    if isinstance(arr, np.ndarray):
                        arr = arr.copy()

                    try:
                        self.data[key].append(arr)
                    except KeyError:
                        self.data[key] = []
                        self.data[key].append(arr)

    def get(self, *args) -> Union[None, np.ndarray, dict]:
        if self.is_empty():
            return None

        if len(args) == 0:
            args = self.data.keys()

        if len(args) == 1:
            return np.array(self.data[args[0]])
        else:
            return {key: np.array(self.data[key]) for key in args}

    def keys(self):
        return self.data.keys()

    def save(self, h5file=None, data_group=''):
        logged_data = self.get()
        if logged_data is not None:
            for key, value in logged_data.items():
                h5file[data_group + '/' + key] = value

    def load(self, h5file=None, data_group=''):
        try:
            for key, value in h5file[data_group].items():
                self.data[key] = list(value)
        except KeyError:
            pass


if __name__ == "__main__":
    Timer.test()

