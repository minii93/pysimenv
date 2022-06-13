from abc import ABC

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
from pytictoc import TicToc
from pysimenv.core.util import SimClock, Timer
from pysimenv.core.base import SimObject, StateVariable, BaseFunction, ArrayType


class BaseSystem(SimObject):
    def __init__(self, *args):
        """
        :param args: initial states
        """
        super().__init__()
        self.name = "base_system"
        self.state_var_list = []
        self.state_index = []

        last_index = 0
        if len(args) > 0:
            for initial_state in args:
                var = StateVariable(initial_state)
                index = list(range(last_index, last_index + var.size))
                self.state_var_list.append(var)
                self.state_index.append(index)
                last_index += var.size

        self.state_var_num = len(self.state_var_list)
        self.state_num = last_index

    def set_state(self, *args) -> None:
        """
        :param args: states
        :return: None
        """
        if len(args) > 0:
            for i, state in enumerate(args):
                self.state_var_list[i].apply_state(state)

    # to be implemented
    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def step(self, dt: float, *args, **kwargs) -> None:
        t_0 = self.sim_clock.time

        self.log_timer.forward()
        self.forward(*args, **kwargs)
        for var in self.state_var_list:
            var.rk4_update_1(dt)

        self.sim_clock.apply_time(t_0 + dt/2)
        self.log_timer.forward()
        self.forward(*args, **kwargs)
        for var in self.state_var_list:
            var.rk4_update_2(dt)

        self.forward(*args, **kwargs)
        for var in self.state_var_list:
            var.rk4_update_3(dt)

        self.sim_clock.apply_time(t_0 + dt - 10*self.sim_clock.time_res)
        self.log_timer.forward()
        self.forward(*args, **kwargs)
        for var in self.state_var_list:
            var.rk4_update_4(dt)

        self.sim_clock.apply_time(t_0 + dt)

    def propagate(self, dt: float, time: float, *args, **kwargs):
        assert self.sim_clock is not None, "Attach a sim_clock first!"
        assert self.log_timer is not None, "Attach a log_timer first!"

        iter_num = min(round(time/dt), np.iinfo(np.int32).max)
        for i in range(iter_num):
            to_stop, _ = self.check_stop_condition()
            if to_stop:
                break
            self.step(dt, *args, **kwargs)

        self.log_timer.forward()
        self.forward(*args, **kwargs)

    def history(self, *args):
        """
        :param args: keys
        :return:
        """
        return self.logger.get(*args)

    def default_plot(self, var_keys=None, var_ind_dict=None, var_names_dict=None):
        if var_keys is None:
            var_keys = list(self.logger.keys())
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

        plt.draw()
        plt.pause(0.01)
        return fig_axs

    # to be implemented
    def report(self):
        print("== Report for {} ==".format(self.name))
        return

    def save(self, h5file=None, data_group=''):
        data_group = data_group + '/' + self.name
        self.logger.save(h5file, data_group)

    def load(self, h5file=None, data_group=''):
        data_group = data_group + '/' + self.name
        self.logger.load(h5file, data_group)

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
    def state(self) -> Union[list, np.ndarray]:
        return self._get_state()

    def _get_state(self) -> list:
        states = []
        for var in self.state_var_list:
            states.append(var.state)
        return states

    @property
    def deriv(self) -> Union[list, np.ndarray]:
        return self._get_deriv()

    def _get_deriv(self) -> list:
        derivs = []
        for var in self.state_var_list:
            derivs.append(var.deriv)
        return derivs


class DynSystem(BaseSystem):
    def __init__(self, initial_state: ArrayType, deriv_fun=None, output_fun=None):
        super().__init__(initial_state)
        self.name = 'dyn_system'

        if output_fun is None:
            def output_fun(x):
                return x

        self.initial_state = initial_state
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
        super().reset()
        self.set_state(self.initial_state)

    # override
    def set_state(self, state: ArrayType):
        self.state_var_list[0].apply_state(state)

    # to be implemented
    def derivative(self, t: float, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    # implement
    def forward(self, *args, **kwargs):
        if self.deriv_fun is None:
            deriv = self.derivative(
                self.sim_clock.time,
                self.state_var_list[0].state,
                *args, **kwargs
            )
        else:
            deriv = self.deriv_fun(
                self.sim_clock.time,
                self.state_var_list[0].state,
                *args, **kwargs
            )
        self.state_var_list[0].forward(deriv)

        if self.log_timer.is_event:
            self.logger.append(
                t=self.sim_clock.time,
                x=self.state_var_list[0].state)
            anonymous_values = {'u_' + str(i): value for i, value in enumerate(args)}
            self.logger.append(**anonymous_values)
            self.logger.append(**kwargs)

    # implement
    def _output(self) -> np.ndarray:
        return self.output_fun(self.state_var_list[0].state)

    # override
    def _get_state(self) -> np.ndarray:
        return self.state_var_list[0].state

    # override
    def _get_deriv(self) -> np.ndarray:
        return self.state_var_list[0].deriv

    @staticmethod
    def test():
        print("== Test for DynSystem ==")
        dt = 0.01
        sim_clock = SimClock()
        log_timer = Timer(dt)
        log_timer.attach_sim_clock(sim_clock)
        log_timer.turn_on()

        def deriv_fun(t, x):
            return -1./((1. + t)**2)*x

        model = DynSystem([1.], deriv_fun)
        model.attach_sim_clock(sim_clock)
        model.attach_log_timer(log_timer)

        t = TicToc()
        t.tic()
        model.propagate(dt=dt, time=10.)
        t.toc()
        model.default_plot()

        plt.show()


class TimeInvarDynSystem(DynSystem):
    def __init__(self, initial_state: ArrayType, deriv_fun=None, output_fun=None):
        if output_fun is None:
            def output_fun(x):
                return x
        super().__init__(initial_state, deriv_fun, output_fun)
        self.name = 'time_invar_dyn_system'

    # to be implemented
    def derivative(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass

    # override
    def forward(self, *args, **kwargs):
        if self.deriv_fun is None:
            deriv = self.derivative(
                self.state_var_list[0].state,
                *args, **kwargs
            )
        else:
            deriv = self.deriv_fun(
                self.state_var_list[0].state,
                *args, **kwargs)
        self.state_var_list[0].forward(deriv)

        if self.log_timer.is_event:
            self.logger.append(
                t=self.sim_clock.time,
                x=self.state_var_list[0].state
            )
            anonymous_values = {'u_' + str(i): value for i, value in enumerate(args)}
            self.logger.append(**anonymous_values)
            self.logger.append(**kwargs)

    @staticmethod
    def test():
        print("== Test for TimeInvarDynSystem ==")
        dt = 0.01
        sim_clock = SimClock()
        log_timer = Timer(dt)
        log_timer.attach_sim_clock(sim_clock)
        log_timer.turn_on()

        def deriv_fun(x, u):
            A = np.array([[0., 1.], [-1., -1.]], dtype=np.float32)
            B = np.array([0., 1.], dtype=np.float32)
            return A.dot(x) + B.dot(u)

        model = TimeInvarDynSystem([0., 1.], deriv_fun)
        model.attach_sim_clock(sim_clock)
        model.attach_log_timer(log_timer)

        t = TicToc()
        t.tic()
        model.propagate(dt=dt, time=10., u=1.)
        t.toc()
        model.default_plot()

        plt.show()


class MultiStateDynSystem(BaseSystem):
    def __init__(self, initial_states: Union[list, tuple], deriv_fun=None, output_fun=None):
        """ initial_states: list or tuple (state1, state2, ...) """
        if output_fun is None:
            def output_fun(*args):
                return args
        super(MultiStateDynSystem, self).__init__(*initial_states)
        self.name = "multi_state_dyn_system"
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
        super(MultiStateDynSystem, self).reset()
        self.set_state(*self.initial_states)

    # to be implemented
    def derivative(self, *args, **kwargs) -> Union[tuple]:
        """ implement this method if needed
        args: tuple (state1, state2, ..., input1, ...)
        return: tuple (derivState1, derivState2, ...)
        """
        if self.deriv_fun is None:
            raise NotImplementedError

        return self.deriv_fun(*args, **kwargs)

    # implement
    def forward(self, *args, **kwargs):
        states = self._get_state()
        derivs = self.derivative(*states, *args, **kwargs)

        for i, state_var in enumerate(self.state_var_list):
            state_var.forward(derivs[i])

        if self.log_timer.is_event:
            self.logger.append(t=self.sim_clock.time)
            state_values = {'x_' + str(i): value for i, value in enumerate(states)}
            anonymous_values = {'u_' + str(i): value for i, value in enumerate(args)}
            self.logger.append(**state_values, **anonymous_values, **kwargs)

    # implement
    def _output(self) -> tuple:
        states = self._get_state()
        return self.output_fun(*states)


class MultipleSystem(BaseSystem, ABC):
    def __init__(self):
        super(MultipleSystem, self).__init__()
        self.name = "model"
        self.sim_obj_list = []
        self.sim_obj_num = 0

    def attach_sim_objects(self, sim_obj_list: Union[SimObject, list, tuple]):
        if not (isinstance(sim_obj_list, list) or isinstance(sim_obj_list, tuple)):
            sim_obj_list = [sim_obj_list]

        svi = self.state_var_num  # last state var index
        si = self.state_num  # last state index

        for sim_obj in sim_obj_list:
            if isinstance(sim_obj, SimObject):
                self.sim_obj_list.append(sim_obj)
                self.sim_obj_num += 1

                if isinstance(sim_obj, BaseSystem):
                    self.state_var_list.extend(sim_obj.state_var_list)
                    for index in sim_obj.state_index:
                        new_index = [si + _i for _i in index]
                        self.state_index.append(new_index)

                    svi += sim_obj.state_var_num
                    si += sim_obj.state_num
        self.state_var_num = svi
        self.state_num = si

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


if __name__ == "__main__":
    DynSystem.test()
    TimeInvarDynSystem.test()


