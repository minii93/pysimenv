import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.core.system import MultipleSystem
from pysimenv.multicopter.model import MulticopterDynamic
from pysimenv.multicopter.control import BSControl


class Model(MultipleSystem):
    def __init__(self):
        super(Model, self).__init__()

        # Quadrotor dynamic model
        m = 1.023
        J = np.diag([9.5, 9.5, 1.86])*1e-3

        pos_0 = np.zeros(3)
        vel_0 = np.zeros(3)
        R_iv_0 = np.identity(3)
        omega_0 = np.zeros(3)
        self.quadrotor_dyn = MulticopterDynamic([pos_0, vel_0, R_iv_0, omega_0], m, J)

        # Back-stepping controller
        alpha = np.array([16., 14., 16., 14., 16., 14., 2.5, 0.5, 2.5, 0.5, 2.5, 0.5])
        self.control = BSControl(m=m, J=J, alpha=alpha)

        self.attach_sim_objects([self.quadrotor_dyn, self.control])

    def forward(self):
        sigma_d = np.array([2., 2., -2., np.deg2rad(15.)])
        sigma_d_dot = np.zeros(4)

        u = self.control.forward(dyn=self.quadrotor_dyn, sigma_d=sigma_d, sigma_d_dot=sigma_d_dot)
        self.quadrotor_dyn.forward(u=u)


def main():
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=10., save_history=True)
    model.quadrotor_dyn.plot_path()
    model.quadrotor_dyn.plot_euler_angles()
    model.quadrotor_dyn.default_plot(show=True)


if __name__ == "__main__":
    main()
