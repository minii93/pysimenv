import numpy as np
from typing import Union, Dict
from pysimenv.core.base import SimObject, BaseFunction, ArrayType


class DynSystem(SimObject):
    def __init__(self, initial_states: Dict[str, ArrayType], deriv_fun=None, output_fun=None,
                 interval: Union[int, float] = -1, name='dyn_sys'):
        """ initial_states: dictionary of (state1, state2, ...) """
        super(DynSystem, self).__init__(initial_states=initial_states, interval=interval, name=name)
        self.initial_states = initial_states

        if isinstance(deriv_fun, BaseFunction):
            self.deriv_fun = deriv_fun.evaluate
        else:
            self.deriv_fun = deriv_fun

        if isinstance(output_fun, BaseFunction):
            self.output_fun = output_fun.evaluate
        else:
            self.output_fun = output_fun

    # override
    def _reset(self):
        super(DynSystem, self)._reset()
        self.set_state(**self.initial_states)

    # may be implemented
    def _deriv(self, **kwargs) -> Dict[str, np.ndarray]:
        """ implement this method if needed
        args: dictionary of {name1: state1, name2: state2, ..., input_name1: input1, ...}
        return: dictionary of {name1: derivState1, name2: derivState2, ...)
        """
        if self.deriv_fun is None:
            raise NotImplementedError
        return self.deriv_fun(**kwargs)

    # override
    def forward(self, **kwargs) -> Union[None, np.ndarray, dict]:
        self._timer.forward()
        output = self._forward(**kwargs)

        if self._timer.is_event:
            self._last_output = output

        return self._last_output

    # implement
    def _forward(self, **kwargs) -> Union[None, np.ndarray, dict]:
        states = self._get_states()
        derivs = self._deriv(**states, **kwargs)

        for name, var in self.state_vars.items():
            var.set_deriv(derivs[name])

        self._logger.append(t=self.time, **states, **kwargs)
        return self._output()

    # override
    @property
    def output(self) -> Union[None, np.ndarray, dict]:
        return self._output()

    # may be overridden
    def _output(self) -> Union[None, np.ndarray, dict]:
        if self.output_fun is None:
            return None
        else:
            states = self._get_states()
            return self.output_fun(**states)


class TimeVaryingDynSystem(SimObject):
    def __init__(self, initial_states: Dict[str, ArrayType], deriv_fun=None, output_fun=None,
                 interval: Union[int, float] = -1, name='time_varying_dyn_sys'):
        super(TimeVaryingDynSystem, self).__init__(initial_states=initial_states, interval=interval, name=name)
        self.initial_states = initial_states

        if isinstance(deriv_fun, BaseFunction):
            self.deriv_fun = deriv_fun.evaluate
        else:
            self.deriv_fun = deriv_fun

        if isinstance(output_fun, BaseFunction):
            self.output_fun = output_fun.evaluate
        else:
            self.output_fun = output_fun

    # override
    def _reset(self):
        super(TimeVaryingDynSystem, self).reset()
        self.set_state(**self.initial_states)

    # to be implemented
    def _deriv(self, t, **kwargs) -> Dict[str, np.ndarray]:
        if self.deriv_fun is None:
            raise NotImplementedError
        return self.deriv_fun(t, **kwargs)

    # implement
    def _forward(self, **kwargs) -> Union[None, np.ndarray, dict]:
        states = self._get_states()
        derivs = self._deriv(t=self.time, **states, **kwargs)

        for name, var in self.state_vars.items():
            var.set_deriv(derivs[name])

        self._logger.append(t=self.time, **states, **kwargs)
        return self._output()

    # override
    @property
    def output(self) -> Union[None, np.ndarray, dict]:
        return self._output()

    # may be overridden
    def _output(self) -> Union[None, np.ndarray, dict]:
        if self.output_fun is None:
            return None
        else:
            states = self._get_states()
            return self.output_fun(t=self.time, **states)
