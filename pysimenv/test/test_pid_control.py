import numpy as np
from pysimenv.core.base import SimObject, DynSystem
from pysimenv.common.model import PIDControl
from pysimenv.core.simulator import Simulator


class Model(SimObject):
    def __init__(self):
        super(Model, self).__init__()
        m = 1.
        c = 6.
        k = 9.8696
        A = np.array([[0., 1.], [-k/m, -c/m]])
        B = np.array([[0.], [1./m]])
        C = np.array([[1., 0.]])
        self.dyn_sys = DynSystem(
            initial_states={'x': np.zeros(2)},
            deriv_fun=lambda x, u: {'x': A.dot(x) + B.dot(u)},
            output_fun=lambda x: C.dot(x)
        )
        self.pid_control = PIDControl(k_p=50., k_d=8., k_i=45.)

        self._add_sim_objs([self.dyn_sys, self.pid_control])

    # implement
    def _forward(self, p_d):
        p = self.dyn_sys.output
        e = p_d - p
        u_pid = self.pid_control.forward(e)

        self.dyn_sys.forward(u=u_pid)


def main():
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=5., save_history=True, p_d=np.array([1.]))
    model.dyn_sys.default_plot(show=True)


if __name__ == "__main__":
    main()

