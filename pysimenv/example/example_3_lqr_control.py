import numpy as np
import scipy.linalg as lin
from pysimenv.core.base import StaticObject, SimObject, DynSystem
from pysimenv.core.simulator import Simulator


class ClosedLoopSys(SimObject):
    def __init__(self):
        super(ClosedLoopSys, self).__init__()

        # open-loop system
        zeta = 0.1
        omega = 1.
        A = np.array([[0., 1.], [-omega**2, -2*zeta*omega]])
        B = np.array([[0.], [omega**2]])

        self.linear_sys = DynSystem(
            initial_states={'x': [0., 1.]},
            deriv_fun=lambda x, u: {'x': A.dot(x) + B.dot(u)}
        )

        # controller
        Q = np.identity(2)
        R = np.identity(1)
        P = lin.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R).dot(B.transpose().dot(P))

        self.lqr_control = StaticObject(interval=0.2, eval_fun=lambda x: -K.dot(x))

        self._attach_sim_objs([self.linear_sys, self.lqr_control])

    # implement
    def _forward(self):
        x = self.linear_sys.state('x')
        u_lqr = self.lqr_control.forward(x=x)
        self.linear_sys.forward(u=u_lqr)


def main():
    model = ClosedLoopSys()
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=10., save_history=True)
    model.linear_sys.default_plot(show=True)


if __name__ == "__main__":
    main()
