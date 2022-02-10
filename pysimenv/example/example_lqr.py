import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
from pysimenv.core.base import BaseObject
from pysimenv.core.system import MultipleSystem, TimeInvarDynSystem
from pysimenv.core.simulator import Simulator


class ExampleLQR(MultipleSystem):
    def __init__(self):
        super(ExampleLQR, self).__init__()

        control_interval = -1  # -1 for continuous control, any positive value for ZOH
        zeta = 0.1
        omega = 1
        A = np.array([
            [0., 1.],
            [-omega**2, -2*zeta*omega]
        ], dtype=np.float32)
        B = np.array([[0.], [omega**2]], dtype=np.float32)
        Q = np.array([[1., 0.], [0., 1.]])
        R = np.array([[1.]])

        P = lin.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R).dot(B.transpose().dot(P))

        self.linear_system = TimeInvarDynSystem([0., 1.], lambda x, u: A.dot(x) + B.dot(u))
        self.lqr_gain = K
        self.lqr_control = BaseObject(interval=control_interval, eval_fun=lambda x: -K.dot(x))

        self.attach_sim_objects([self.linear_system, self.lqr_control])

    # implement
    def forward(self):
        x = self.linear_system.state
        u_lqr = self.lqr_control.forward(x)
        self.linear_system.forward(u=u_lqr)


if __name__ == "__main__":
    print("== Test for an example LQR model ==")
    model = ExampleLQR()
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=10., save_history=True)
    model.linear_system.default_plot()

    plt.show()
