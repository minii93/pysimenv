import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.core.system import MultipleSystem
from pysimenv.multicopter.model import MulticopterDynamic, QuadXThrustModel
from pysimenv.multicopter.control import OAFControl


class Model(MultipleSystem):
    def __init__(self):
        super(Model, self).__init__()

        # Quadrotor dynamic model
        m = 1.023
        J = np.diag([9.5, 9.5, 1.86])*0.001
        d_v = np.array([1.2, 1.2, 1.6])
        d_omega = np.array([0.06, 0.06, 0.1])

        p_0 = np.array([0., 0., -5.])
        v_0 = np.zeros(3)
        R_ib_0 = np.identity(3)
        omega_0 = np.zeros(3)

        self.quadrotor_dyn = MulticopterDynamic([p_0, v_0, R_ib_0, omega_0], m, J,
                                                D_v=np.diag(d_v), D_omega=np.diag(d_omega))

        # Quadrotor thrust model (QuadX configuration)
        self.quadrotor_thrust = QuadXThrustModel(d_phi=0.15, d_theta=0.15, c_tau_f=0.0196)

        # Controller
        G = self.quadrotor_thrust.R_u
        self.control = OAFControl(m=m, J=J, d_v=d_v, d_omega=d_omega, G=G,
                                  K_p=np.array([1.24, 1.24, 1.24, 225., 225.]),
                                  K_d=np.array([1., 1., 1., 30., 30.]))

        self.attach_sim_objects([
            self.quadrotor_dyn, self.control
        ])

    def forward(self):
        p_d = np.array([2., 2., 0.])
        p_d_dot = np.zeros(3)
        p_d_2dot = np.zeros(3)

        w_r = self.control.forward(
            dyn=self.quadrotor_dyn, p_d=p_d, p_d_dot=p_d_dot, p_d_2dot=p_d_2dot,
            fault_ind=2
        )
        w = np.array([w_r[0], w_r[1], 0., w_r[2]])
        u = self.quadrotor_thrust.convert(w)
        self.quadrotor_dyn.forward(u=u)


def main():
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=20., save_history=True)
    model.quadrotor_dyn.plot_path()
    model.quadrotor_dyn.plot_euler_angles()
    model.control.plot_desired_att()
    model.quadrotor_dyn.default_plot(show=True)


if __name__ == "__main__":
    main()
