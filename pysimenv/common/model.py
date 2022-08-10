import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Tuple, Optional
from pysimenv.core.base import SimObject, StaticObject, ArrayType
from pysimenv.core.system import DynObject, MultipleSystem, DynSystem


class SignalGenerator(StaticObject):
    def __init__(self, shaping_fun, interval: Union[int, float] = -1):
        super(SignalGenerator, self).__init__(interval=interval)
        self.shaping_fun = shaping_fun

    # implementation
    def _forward(self) -> np.ndarray:
        return self.shaping_fun(self.time)


class Scope(StaticObject):
    def __init__(self):
        super(Scope, self).__init__()

    # implementation
    def _forward(self, **kwargs):
        self._logger.append(t=self.time, **kwargs)

    def plot(self, show: bool = False):
        var_names = list(self._logger.keys())
        var_names.remove('t')

        t = self.history('t')
        for var_name in var_names:
            var = self.history(var_name)

            fig, ax = plt.subplots()
            for i in range(var.shape[1]):
                ax.plot(t, var[:, i], label=var_name + "_" + str(i))
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Value")
                ax.grid()
                ax.legend()
            fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)


class FeedbackControl(MultipleSystem):
    def __init__(self, system: DynObject, control: StaticObject):
        super(FeedbackControl, self).__init__()
        self.system = system
        self.control = control

        self.attach_sim_objects([system, control])

    # implement
    def _forward(self, r: Optional[np.ndarray] = None):
        y = self.system.output
        if r is None:
            u_fb = self.control.forward(y)
        else:
            u_fb = self.control.forward(y, r)
        self.system.forward(u=u_fb)


class Sequential(MultipleSystem):
    def __init__(self, obj_list: Union[List[SimObject], Tuple[SimObject]]):
        super(Sequential, self).__init__()
        assert len(obj_list) > 0, "(sequential) Invalid obj_list!"
        self.obj_list = list(obj_list)
        self.first_obj = obj_list[0]
        self.other_obj_list = obj_list[1:]

        self.attach_sim_objects(obj_list)

    # implement
    def _forward(self, **kwargs):
        self.first_obj.forward(**kwargs)
        out = self.first_obj.output
        for obj in self.other_obj_list:
            if out is None:
                obj.forward()
                out = obj.output
            elif isinstance(out, dict):
                obj.forward(**out)
                out = obj.output
            else:
                obj.forward(u=out)
                out = obj.output
        return out


class FlatEarthEnv(object):
    grav_accel = 9.805


class LinSys(DynSystem):
    def __init__(self, x_0: ArrayType, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None):
        super(LinSys, self).__init__(initial_states={'x': x_0})
        self.A = A
        self.B = B
        if C is None:
            C = np.eye(self.state_dim, dtype=np.float32)
        self.C = C

    # implement
    def _deriv(self, x: np.ndarray, u: Union[None, np.ndarray] = None):
        if self.B is None or u is None:
            x_dot = self.A.dot(x)
        else:
            x_dot = self.A.dot(x) + self.B.dot(u)

        return {'x': x_dot}

    # implement
    def _output(self) -> np.ndarray:
        return self.C.dot(self.state('x'))


class FirstOrderLinSys(LinSys):
    def __init__(self, x_0: ArrayType, tau: float):
        A = np.array([[-1./tau]])
        B = np.array([[1./tau]])

        super(FirstOrderLinSys, self).__init__(x_0, A, B)
        self.tau = tau

    # implement
    def _output(self) -> Union[None, tuple, np.ndarray]:
        return self.state('x')[0]


class SecondOrderLinSys(LinSys):
    def __init__(self, x_0: ArrayType, zeta: float, omega: float):
        A = np.array([
            [0., 1.],
            [-omega**2, -2*zeta*omega]
        ], dtype=np.float32)
        B = np.array([[0.], [omega**2]], dtype=np.float32)

        super(SecondOrderLinSys, self).__init__(x_0, A, B)
        self.zeta = zeta
        self.omega = omega

    # implement
    def _output(self) -> Union[None, tuple, np.ndarray]:
        return self.state('x')[0]


class Integrator(DynSystem):
    def __init__(self, initial_state: Union[ArrayType]):
        def deriv_fun(x: np.ndarray, u: Union[float, np.ndarray]):
            if isinstance(u, float):
                u = np.array([u])
            return {'x': u}

        super(Integrator, self).__init__({'x': initial_state}, deriv_fun)


class Differentiator(StaticObject):
    def __init__(self):
        super(Differentiator, self).__init__()
        self.u_prev = None
        self.t_prev = None

    # implement
    def _forward(self, u: np.ndarray):
        if self.u_prev is None:
            deriv = np.zeros_like(u)
        else:
            deriv = (u - self.u_prev)/(self.time - self.t_prev)

        if self._sim_clock.major_time_step:
            self.u_prev = u.copy()
            self.t_prev = self.time

        return deriv


class PIDControl(DynObject):
    def __init__(self, k_p: Union[float, np.ndarray], k_i: Union[float, np.ndarray], k_d: Union[float, np.ndarray],
                 windup_limit=None,
                 interval: Union[int, float] = -1):
        super(PIDControl, self).__init__(initial_states={'e_i': np.zeros_like(k_p)}, interval=interval)
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.windup_limit = windup_limit
        if self.windup_limit:
            self.state_vars['e_i'].attach_correction_fun(self.clip_windup)

        self.e_prev = None
        self.t_prev = None

    # implement
    def _forward(self, e: np.ndarray) -> np.ndarray:
        self.state_vars['e_i'].set_deriv(deriv=e)

        e_i = self.state('e_i')
        e_d = np.zeros_like(e) if self.e_prev is None else (e - self.e_prev)/(self.time - self.t_prev)
        u_pid = self.k_p*e + self.k_i*e_i + self.k_d*e_d

        if self._sim_clock.major_time_step:
            self.e_prev = e.copy()
            self.t_prev = self.time

        return u_pid

    def clip_windup(self, e_i: np.ndarray) -> np.ndarray:
        return np.clip(e_i, self.windup_limit, -self.windup_limit)
