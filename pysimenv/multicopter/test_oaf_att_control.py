import numpy as np
from pysimenv.core.simulator import Simulator
from pysimenv.core.system import MultipleSystem
from pysimenv.common.orientation import euler_angles_to_rotation
from pysimenv.multicopter.model import MulticopterDynamic, QuadXThrustModel
from pysimenv.multicopter.control import OAFAttControl


class Model(MultipleSystem):
    def __init__(self):
        super(Model, self).__init__()

        # Quadrotor dynamic model
        m = 1.023
        J = np.diag([9.5, 9.5, 1.86])*0.001
        D_v = np.diag([1.2, 1.2, 1.6])
        D_omega = np.diag([0.06, 0.06, 0.1])

        pos_0 = np.array([0., 0., -5.])
        vel_0 = np.zeros(3)
        eta_0 = np.deg2rad([15., -10., 0.])
        R_bi_0 = euler_angles_to_rotation(eta_0)
        R_ib_0 = R_bi_0.transpose()
        omega_0 = np.zeros(3)
        self.quadrotor_dyn = MulticopterDynamic([pos_0, vel_0, R_ib_0, omega_0], m, J,
                                                D_v=D_v, D_omega=D_omega)

        # Quadrotor thrust model (QuadX configuration)
        self.quadrotor_thrust = QuadXThrustModel(d_phi=0.15, d_theta=0.15, c_tau_f=0.0196)

        # Controller
        G = self.quadrotor_thrust.R_u
        self.att_control = OAFAttControl(m=m, J=J, D_v=D_v, D_omega=D_omega,
                                         G=G, K_1=np.array([12., 12., 2.5]), K_2=np.array([10., 10., 0.5]))

        self.attach_sim_objects([
            self.quadrotor_dyn, self.att_control
        ])

    def forward(self):
        zeta_d = np.array([0., 0., -2.])
        zeta_d_dot = np.zeros(3)
        zeta_d_2dot = np.zeros(3)

        w_r = self.att_control.forward(
            dyn=self.quadrotor_dyn, zeta_d=zeta_d, zeta_d_dot=zeta_d_dot, zeta_d_2dot=zeta_d_2dot,
            fault_ind=2
        )
        w = np.array([w_r[0], w_r[1], 0., w_r[2]])
        u = self.quadrotor_thrust.convert(w)
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

