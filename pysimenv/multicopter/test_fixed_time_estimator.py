import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.system import MultipleSystem
from pysimenv.core.simulator import Simulator
from pysimenv.multicopter.model import MulticopterDynamic, QuadXThrustModel, QuadXMixer, ActuatorFault
from pysimenv.multicopter.control import QuaternionPosControl, QuaternionAttControl
from pysimenv.multicopter.estimator import FixedTimeFaultEstimator


class Model(MultipleSystem):
    def __init__(self):
        super(Model, self).__init__()

        # Quadrotor dynamic model
        m = 1.212
        J = np.diag([1.0, 8.2, 1.48])*0.01

        pos_0 = np.array([0., 1., -1.])
        vel_0 = np.zeros(3)
        R_iv_0 = np.identity(3)
        omega_0 = np.zeros(3)
        self.quadrotor_dyn = MulticopterDynamic([pos_0, vel_0, R_iv_0, omega_0], m, J)

        # Quadrotor thrust model (QuadX configuration)
        d_phi = 0.15
        d_theta = 0.13
        c_tau_f = 0.02
        self.quadrotor_thrust = QuadXThrustModel(d_phi, d_theta, c_tau_f)

        # Quadrotor actuator mixer
        self.quadrotor_mixer = QuadXMixer(d_phi, d_theta, c_tau_f)

        # Actuator fault model
        self.actuator_fault = ActuatorFault(
            t_list=[0., 10.],
            alp_list=[
                np.array([1., 1., 1., 1.]),
                np.array([0.7, 1., 1., 1.])
            ],
            rho_list=[
                np.zeros(4),
                np.zeros(4)
            ]
        )

        # Baseline controller
        K_att = QuaternionAttControl.gain(
            Q=np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2,
            R=1e-4*np.identity(3)
        )
        K_pos = QuaternionPosControl.gain(
            Q=np.diag([1., 1., 1., 0.1, 0.1, 0.1])**2,
            R=np.identity(3)
        )
        self.att_control = QuaternionAttControl(J, K_att)
        self.pos_control = QuaternionPosControl(m, K_pos)

        # Fixed-Time Fault Estimator
        v_z_0 = vel_0[2]
        p_0, q_0, r_0 = omega_0[:]
        z_1_0 = np.array([v_z_0, p_0, q_0, r_0])
        z_2_0 = np.zeros(4)

        self.estimator = FixedTimeFaultEstimator(
            initial_states=[z_1_0, z_2_0],
            alpha=0.733, beta=1.285, k_1=-18., k_2=-100.,
            m=m, J=J, R_u=self.quadrotor_thrust.R_u
        )

        self.attach_sim_objects([
            self.quadrotor_dyn, self.actuator_fault, self.att_control, self.pos_control, self.estimator])

    def forward(self, p_d: np.ndarray, v_d: np.ndarray = np.zeros(3)) -> None:
        p = self.quadrotor_dyn.pos
        v = self.quadrotor_dyn.vel
        q = self.quadrotor_dyn.quaternion
        omega = self.quadrotor_dyn.ang_vel

        # position control
        F_m, q_d, omega_d = self.pos_control.forward(p, v, p_d, v_d)

        # attitude control
        M = self.att_control.forward(q, omega, q_d, omega_d)

        # Mixing
        f_s = self.quadrotor_mixer.convert(np.array([F_m, M[0], M[1], M[2]]))

        # Fault estimation
        v_z = v[2]
        eta = self.quadrotor_dyn.euler_ang
        self.estimator.forward(
            x=np.array([v_z, omega[0], omega[1], omega[2]]), eta=eta, f_s=f_s)
        delta_hat = self.estimator.delta_hat

        f_s_star = self.actuator_fault.forward(f_s)
        u = self.quadrotor_thrust.convert(f_s_star)
        self.quadrotor_dyn.forward(u)

        # true uncertainty
        m = self.quadrotor_dyn.m
        J = self.quadrotor_dyn.J
        J_x, J_y, J_z = J[0, 0], J[1, 1], J[2, 2]
        R_u = self.quadrotor_thrust.R_u

        phi, theta = eta[0:2]
        B = np.diag([
            np.cos(phi)*np.cos(theta)/m, 1./J_x, 1./J_y, 1./J_z
        ])
        delta = B.dot(R_u.dot(f_s_star - f_s))

        self.logger.append(t=self.time, f_s=f_s, f_s_star=f_s_star,
                           delta=delta, delta_hat=delta_hat)

    def plot_actuator_log(self, show=False):
        t = self.history('t')
        f_s = self.history('f_s')
        f_s_star = self.history('f_s_star')

        fig, ax = plt.subplots(4, 1, figsize=(8, 6))
        ylabel_list = ["Motor 1", "Motor 2", "Motor 3", "Motor 4"]
        for i in range(4):
            ax[i].plot(t, f_s[:, i], label="Command")
            ax[i].plot(t, f_s_star[:, i], label="Actual", linestyle='--')
            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel(ylabel_list[i])
            ax[i].grid()
            ax[i].legend()
        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)

    def plot_uncertainty(self, show=False):
        t = self.history('t')
        delta = self.history('delta')
        delta_hat = self.history('delta_hat')

        fig, ax = plt.subplots(4, 1, figsize=(8, 6))
        ylabel_list = ["Delta v_z", "Delta p", "Delta q", "Delta r"]
        for i in range(4):
            ax[i].plot(t, delta[:, i], label="Actual")
            ax[i].plot(t, delta_hat[:, i], label="Estimated", linestyle='--')
            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel(ylabel_list[i])
            ax[i].grid()
            ax[i].legend()
        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)


def main():
    p_d = np.array([0., 0., 1.])
    v_d = np.array([0., 0., 0.])
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(0.01, 20., True, p_d, v_d)
    # model.quadrotor_dyn.default_plot(show=False)
    model.plot_actuator_log(show=False)
    model.plot_uncertainty(show=True)


if __name__ == "__main__":
    main()
