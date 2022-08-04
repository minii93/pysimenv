import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from typing import Union, Tuple, Dict, Optional
from pysimenv.core.util import SimClock, Timer
from pysimenv.core.base import SimObject, StateVariable, BaseFunction, ArrayType


class BaseSystem(SimObject):
    def __init__(self, **kwargs):
        """
        :param kwargs: initial states
        """
        super().__init__()
        self.name = "base_system"
        self.state_vars: Dict[str, StateVariable] = dict()

        state_dim = 0
        if len(kwargs) > 0:
            for name, initial_state in kwargs.items():
                var = StateVariable(initial_state)
                self.state_vars[name] = var
                state_dim += var.size

        self.num_state_var = len(self.state_vars)
        self.state_dim = state_dim

    def set_state(self, **kwargs) -> None:
        """
        :param kwargs: states
        :return: None
        """
        if len(kwargs) > 0:
            for name, state in kwargs.items():
                self.state_vars[name].apply_state(state)

    # to be implemented
    def forward(self, **kwargs) -> None:
        raise NotImplementedError

    def step(self, dt: float, **kwargs) -> None:
        t_0 = self._sim_clock.time

        self._log_timer.forward()
        self.forward(**kwargs)
        for var in self.state_vars.values():
            var.rk4_update_1(dt)

        self._sim_clock.apply_time(t_0 + dt / 2)
        self._log_timer.forward()
        self.forward(**kwargs)
        for var in self.state_vars.values():
            var.rk4_update_2(dt)

        self.forward(**kwargs)
        for var in self.state_vars.values():
            var.rk4_update_3(dt)

        self._sim_clock.apply_time(t_0 + dt - 10 * self._sim_clock.time_res)
        self._log_timer.forward()
        self.forward(**kwargs)
        for var in self.state_vars.values():
            var.rk4_update_4(dt)

        self._sim_clock.apply_time(t_0 + dt)

    def propagate(self, dt: float, time: float, **kwargs):
        assert self._sim_clock is not None, "Attach a sim_clock first!"
        assert self._log_timer is not None, "Attach a log_timer first!"

        iter_num = min(round(time/dt), np.iinfo(np.int32).max)
        for i in range(iter_num):
            to_stop, _ = self.check_stop_condition()
            if to_stop:
                break
            self.step(dt, **kwargs)

        self._log_timer.forward()
        self.forward(**kwargs)

    def default_plot(self, show=False, var_keys=None, var_ind_dict=None, var_names_dict=None):
        if var_keys is None:
            var_keys = list(self._logger.keys())
            var_keys.remove('t')
        if var_ind_dict is None:
            var_ind_dict = dict()
        if var_names_dict is None:
            var_names_dict = dict()

        fig_axs = dict()
        time_list = self.history('t')

        for var_key in var_keys:
            var_list = self.history(var_key)
            if var_list.ndim == 1:
                var_list = np.reshape(var_list, var_list.shape + (1, ))

            if var_key in var_ind_dict:
                ind = var_ind_dict[var_key]
            else:
                ind = list(range(var_list.shape[1]))

            if var_key in var_names_dict:
                names = var_names_dict[var_key]
            else:
                names = [var_key + "_" + str(k) for k in ind]

            subplot_num = len(ind)
            fig, ax = plt.subplots(subplot_num, 1)
            fig_axs[var_key] = {
                'fig': fig,
                'ax': ax
            }
            for i in range(subplot_num):
                if subplot_num == 1:
                    ax_ = ax
                else:
                    ax_ = ax[i]
                ax_.plot(time_list, var_list[:, ind[i]], label="Actual")
                ax_.set_xlabel("Time (s)")
                ax_.set_ylabel(names[i])
                ax_.grid()
                ax_.legend()
            fig.suptitle(var_key)

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)
        return fig_axs

    # to be implemented
    def report(self):
        pass

    def save(self, h5file=None, data_group=''):
        data_group = data_group + '/' + self.name
        self._logger.save(h5file, data_group)

    def load(self, h5file=None, data_group=''):
        data_group = data_group + '/' + self.name
        self._logger.load(h5file, data_group)

    def save_log_file(self, save_dir=None):
        if save_dir is None:
            save_dir = './data/'
        os.makedirs(save_dir, exist_ok=True)
        file = h5py.File(save_dir + 'log.hdf5', 'w')
        self.save(file)
        file.close()

    def load_log_file(self, save_dir=None):
        if save_dir is None:
            save_dir = './data/'
        file = h5py.File(save_dir + 'log.hdf5', 'r')
        self.load(file)
        file.close()

    @property
    def state(self) -> Dict[str, np.ndarray]:
        return self._get_states()

    def _get_states(self) -> Dict[str, np.ndarray]:
        states = dict()
        for name, var in self.state_vars.items():
            states[name] = var.state
        return states

    @property
    def deriv(self) -> Dict[str, np.ndarray]:
        return self._get_deriv()

    def _get_deriv(self) -> Dict[str, np.ndarray]:
        derivs = dict()
        for name, var in self.state_vars.items():
            derivs[name] = var.deriv
        return derivs


class DynSystem(BaseSystem):
    def __init__(self, initial_states: Dict[str, ArrayType], deriv_fun=None, output_fun=None):
        """ initial_states: dictionary of (state1, state2, ...) """
        if output_fun is None:
            def output_fun(**kwargs):
                return None

        super(DynSystem, self).__init__(**initial_states)
        self.name = "dyn_system"
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
    def reset(self):
        super(DynSystem, self).reset()
        self.set_state(**self.initial_states)

    # to be implemented
    def derivative(self, **kwargs) -> Dict[str, np.ndarray]:
        """ implement this method if needed
        args: dictionary of {name1: state1, name2: state2, ..., input_name1: input1, ...}
        return: dictionary of {name1: derivState1, name2: derivState2, ...)
        """
        if self.deriv_fun is None:
            raise NotImplementedError
        return self.deriv_fun(**kwargs)

    # implement
    def forward(self, **kwargs):
        states = self._get_states()
        derivs = self.derivative(**states, **kwargs)

        for name, var in self.state_vars.items():
            var.forward(derivs[name])

        if self._log_timer.is_event:
            self._logger.append(t=self.time, **states, **kwargs)

    # implement
    def _output(self) -> Optional[np.ndarray]:
        states = self._get_states()
        return self.output_fun(**states)


class TimeVaryingDynSystem(BaseSystem):
    def __init__(self, initial_states: Dict[str, ArrayType], deriv_fun=None, output_fun=None):
        if output_fun is None:
            def output_fun(t, **kwargs):
                return None

        super(TimeVaryingDynSystem, self).__init__(**initial_states)
        self.name = "time_varying_dyn_system"
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
    def reset(self):
        super(TimeVaryingDynSystem, self).reset()
        self.set_state(**self.initial_states)

    # to be implemented
    def derivative(self, t, **kwargs) -> Dict[str, np.ndarray]:
        if self.deriv_fun is None:
            raise NotImplementedError
        return self.deriv_fun(t, **kwargs)

    # implement
    def forward(self, **kwargs):
        states = self._get_states()
        derivs = self.derivative(t=self.time, **states, **kwargs)

        for name, var in self.state_vars.items():
            var.forward(derivs[name])

        if self._log_timer.is_event:
            self._logger.append(t=self.time, **states, **kwargs)

    # implement
    def _output(self) -> Optional[np.ndarray]:
        states = self._get_states()
        return self.output_fun(t=self.time, **states)


class MultipleSystem(BaseSystem, ABC):
    def __init__(self):
        super(MultipleSystem, self).__init__()
        self.name = "model"
        self.sim_obj_list = []
        self.sim_obj_num = 0

    def attach_sim_objects(self, sim_obj_list: Union[SimObject, list, tuple]):
        if isinstance(sim_obj_list, SimObject):
            sim_obj_list = [sim_obj_list]

        for sim_obj in sim_obj_list:
            if not isinstance(sim_obj, SimObject):
                continue

            self.sim_obj_list.append(sim_obj)
            self.sim_obj_num += 1
            if isinstance(sim_obj, BaseSystem):
                for name, var in sim_obj.state_vars.items():
                    modified_name = 'sub' + str(self.sim_obj_num - 1) + '_' + name
                    self.state_vars[modified_name] = var

                self.num_state_var += sim_obj.num_state_var
                self.state_dim += sim_obj.state_dim

    # override
    def attach_sim_clock(self, sim_clock: SimClock):
        super(MultipleSystem, self).attach_sim_clock(sim_clock)
        for sim_obj in self.sim_obj_list:
            sim_obj.attach_sim_clock(sim_clock)

    # override
    def attach_log_timer(self, log_timer: Timer):
        super(MultipleSystem, self).attach_log_timer(log_timer)
        for sim_obj in self.sim_obj_list:
            sim_obj.attach_log_timer(log_timer)

    # override
    def reset(self):
        super(MultipleSystem, self).reset()
        for sim_obj in self.sim_obj_list:
            sim_obj.reset()

    # implement
    def check_stop_condition(self) -> Tuple[bool, list]:
        to_stop_list = []
        for sim_obj in self.sim_obj_list:
            to_stop_list.append(sim_obj.check_stop_condition())

        to_stop = np.array(to_stop_list, dtype=bool).any()
        if to_stop:
            self.flag = []
            for sim_obj in self.sim_obj_list:
                self.flag.append(sim_obj.flag)

        return to_stop, self.flag

    # implement
    def save(self, h5file=None, data_group=''):
        super().save(h5file, data_group)
        for sim_obj in self.sim_obj_list:
            if isinstance(sim_obj, BaseSystem):
                sim_obj.save(h5file, data_group + '/' + self.name)

    # implement
    def load(self, h5file=None, data_group=''):
        super().load(h5file, data_group)
        for sim_obj in self.sim_obj_list:
            if isinstance(sim_obj, BaseSystem):
                sim_obj.load(h5file, data_group + '/' + self.name)
