import numpy as np
from typing import Union, List, Tuple, Optional
from pysimenv.core.base import SimObject, StaticObject, ArrayType
from pysimenv.core.system import BaseSystem, MultipleSystem, DynSystem


class SignalGenerator(StaticObject):
    def __init__(self, shaping_fun, interval: Union[int, float] = -1):
        super(SignalGenerator, self).__init__(interval=interval)
        self.shaping_fun = shaping_fun

    # implementation
    def evaluate(self):
        return self.shaping_fun(self._sim_clock.time)


class FeedbackControl(MultipleSystem):
    def __init__(self, system: BaseSystem, control: StaticObject):
        super(FeedbackControl, self).__init__()
        self.system = system
        self.control = control

        self.attach_sim_objects([system, control])

    # implement
    def forward(self, r: Optional[np.ndarray] = None):
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
    def forward(self, **kwargs):
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

    def derivative(self, x: np.ndarray, u: Union[None, np.ndarray] = None):
        if self.B is None or u is None:
            x_dot = self.A.dot(x)
        else:
            x_dot = self.A.dot(x) + self.B.dot(u)

        return {'x': x_dot}

    # implement
    def _output(self) -> np.ndarray:
        return self.C.dot(self.state['x'])


class FirstOrderLinSys(LinSys):
    def __init__(self, x_0: ArrayType, tau: float):
        A = np.array([[-1./tau]])
        B = np.array([[1./tau]])

        super(FirstOrderLinSys, self).__init__(x_0, A, B)
        self.tau = tau

    def _output(self) -> Union[None, tuple, np.ndarray]:
        return self.state['x'][0]


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

    def _output(self) -> Union[None, tuple, np.ndarray]:
        return self.state['x'][0]


class Integrator(DynSystem):
    def __init__(self, initial_state: Union[ArrayType]):
        def deriv_fun(x: np.ndarray, u: Union[float, np.ndarray]):
            if isinstance(u, float):
                u = np.array([u])
            return {'x': u}

        super(Integrator, self).__init__({'x': initial_state}, deriv_fun)
