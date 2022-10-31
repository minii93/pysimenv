import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.base import SimObject
from pysimenv.core.simulator import Simulator
from pysimenv.multicopter.base import MulticopterDyn
from pysimenv.multicopter.model import QuadXEffector, QuadXMixer, ActuatorFault
from pysimenv.multicopter.control import QuaternionPosControl, QuaternionAttControl
from pysimenv.multicopter.estimator import FixedTimeFaultEstimator
from pysimenv.common.model import FlatEarthEnv, SignalGenerator


class ISMC(SimObject):
    """
    Integral sliding mode control
    """
    def __init__(self, x_b_0: np.ndarray, N: np.ndarray, eps_1: float, eps_2: float,
                 J: np.ndarray, m: float, **kwargs):
        super(ISMC, self).__init__(**kwargs)
        self._add_state_vars(x_b=x_b_0)  # initialize the baseline state
        self.N = N.copy()
        self.eps_1 = eps_1
        self.eps_2 = eps_2
        self.J = J.copy()
        self.m = m
        self.s = np.zeros(4)
        self.u_f = np.zeros(4)

    def _forward(self, dyn: MulticopterDyn, x_d: np.ndarray, u_b: np.ndarray, delta_hat: np.ndarray) -> np.ndarray:
        """
        :param x_d: desired state
        :param x: actual state (v_z, p, q, r)
        :param eta: Euler angles (phi, theta, psi)
        :param u_b: baseline control input
        :param delta_hat: estimated uncertainty
        :return:
        """
        # update the baseline state
        phi, theta = dyn.euler_ang[0:2]
        p, q, r = dyn.ang_vel[:]

        J_x, J_y, J_z = self.J[0, 0], self.J[1, 1], self.J[2, 2]
        f = np.array([
            FlatEarthEnv.grav_accel,
            (J_y - J_z)/J_x*q*r,
            (J_z - J_x)/J_y*p*r,
            (J_x - J_y)/J_z*p*q
        ])
        B = np.diag([
            -np.cos(phi)*np.cos(theta)/self.m, 1./J_x, 1./J_y, 1./J_z
        ])
        x_b_dot = f + B.dot(u_b)
        self.state_vars['x_b'].set_deriv(deriv=x_b_dot)

        # calculate the control input
        x_b = self.state('x_b')
        s = self.N.dot(x_d - x_b)
        sigma = 2/np.pi*np.arctan(np.linalg.norm(s))*s
        self.u_f = -np.linalg.solve(np.matmul(self.N, B), self.N.dot(delta_hat) + self.eps_1*sigma + self.eps_2*s)

        self._logger.append(t=self.time, s=s)
        return self.u_f.copy()

    def plot_sliding_value(self, show=False):
        t = self.history('t')
        s = self.history('s')

        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(4):
            ax.plot(t, s[:, i], label="s_" + str(i))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Sliding value")
        ax.grid()
        ax.legend()

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)


class Model(SimObject):
    def __init__(self):
        super(Model, self).__init__()

        # Trajectory
        T = 15.  # (s)
        self.pos_trajectory = SignalGenerator(
            shaping_fun=lambda t: np.array([np.sin(2*np.pi*t/T), np.cos(2*np.pi*t/T), -1.])
        )
        self.vel_trajectory = SignalGenerator(
            shaping_fun=lambda t: np.array([
                2*np.pi/T*np.cos(2*np.pi*t/T), -2*np.pi/T*np.sin(2*np.pi*t/T), 0.
            ])
        )

        # Quadrotor dynamic model
        model_params = {'m': 1.212, 'J': np.diag([1.0, 8.2, 1.48])*0.01}

        self.dyn = MulticopterDyn(p_0=np.array([0., 1., -1.]), v_0=np.zeros(3), R_0=np.identity(3), omega_0=np.zeros(3),
                                  **model_params, name='dynamic model')

        # Quadrotor thrust model (QuadX configuration)
        d_phi = 0.15
        d_theta = 0.13
        c_tau_f = 0.02
        self.effector = QuadXEffector(d_phi, d_theta, c_tau_f)

        # Quadrotor actuator mixer
        self.mixer = QuadXMixer(d_phi, d_theta, c_tau_f)

        # Actuator fault model
        self.actuator_fault = ActuatorFault(
            t_list=[0., 12., 25., 38.],
            alp_list=[
                np.array([1., 1., 1., 1.]),
                np.array([0.7, 1., 1., 1.]),
                np.array([0.7, 1., 0.6, 1.]),
                np.array([0.7, 0.8, 0.6, 0.9])
            ],
            rho_list=[
                np.zeros(4),
                np.zeros(4),
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
        self.att_control = QuaternionAttControl(model_params['J'], K_att)
        self.pos_control = QuaternionPosControl(model_params['m'], K_pos)

        # Fixed-Time Fault Estimator
        v_z_0 = 0.
        p_0, q_0, r_0 = 0., 0., 0.
        z_1_0 = np.array([v_z_0, p_0, q_0, r_0])
        z_2_0 = np.zeros(4)

        self.estimator = FixedTimeFaultEstimator(
            initial_states=[z_1_0, z_2_0],
            alpha=0.733, beta=1.285, k_1=20., k_2=100., **model_params
        )

        # Integral Sliding Mode Controller
        self.ismc = ISMC(
            x_b_0=np.zeros(4), N=np.diag([1., 1., 1., 1.]), eps_1=0.5, eps_2=2.5, **model_params)

        self._add_sim_objs([
            self.pos_trajectory, self.vel_trajectory,
            self.dyn, self.actuator_fault, self.att_control, self.pos_control,
            self.estimator, self.ismc])

    def forward(self) -> None:
        p_d = self.pos_trajectory.forward()
        v_d = self.vel_trajectory.forward()

        # Baseline control (position)
        F_m, q_d, omega_d = self.pos_control.forward(self.dyn, p_d, v_d)

        # Baseline control (attitude)
        M = self.att_control.forward(self.dyn, q_d, omega_d)
        u_b = np.array([F_m, M[0], M[1], M[2]])

        # Uncertainty compensation
        v_z_d = v_d[2]
        x_d = np.array([v_z_d, omega_d[0], omega_d[1], omega_d[2]])

        delta_hat = self.estimator.delta_hat
        u_f = self.ismc.forward(self.dyn, x_d, u_b, delta_hat)
        u = u_b + u_f

        # actuator fault
        f_s = self.mixer.convert(f=u[0], tau=u[1:4])
        f_s_star = self.actuator_fault.forward(f_s)
        u_star = self.effector.convert(f_s_star)
        self.dyn.forward(**u_star)

        # Fault estimation
        v_z = self.dyn.vel[2]
        p, q, r = self.dyn.ang_vel[:]
        x = np.array([v_z, p, q, r])
        eta = self.dyn.euler_ang

        self.estimator.forward(x=x, eta=eta, u=u)

        # true uncertainty
        m = self.dyn.m
        J = self.dyn.J
        J_x, J_y, J_z = J[0, 0], J[1, 1], J[2, 2]

        phi, theta = eta[0:2]
        B = np.diag([
            -np.cos(phi)*np.cos(theta)/m, 1./J_x, 1./J_y, 1./J_z
        ])
        f_star = u_star['f']
        tau_star = u_star['tau']
        delta = B.dot(np.array([f_star, tau_star[0], tau_star[1], tau_star[2]]) - u)

        self._logger.append(t=self.time, f_s=f_s, f_s_star=f_s_star,
                            delta=delta, delta_hat=delta_hat, p_d=p_d, p=self.dyn.pos)

    def plot_actuator_log(self, show=False):
        t = self.history('t')
        f_s = self.history('f_s')
        f_s_star = self.history('f_s_star')

        fig, ax = plt.subplots(4, 1, figsize=(8, 6))
        ylabels = ["Motor 1 (N)", "Motor 2 (N)", "Motor 3 (N)", "Motor 4 (N)"]
        for i in range(4):
            ax[i].plot(t, f_s[:, i], label="Command")
            ax[i].plot(t, f_s_star[:, i], label="Actual", linestyle='--')
            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel(ylabels[i])
            ax[i].grid()
            ax[i].legend()
        fig.suptitle("Actuator status")
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
        ylabels = ["Delta v_z", "Delta p", "Delta q", "Delta r"]
        for i in range(4):
            ax[i].plot(t, delta[:, i], label="Actual")
            ax[i].plot(t, delta_hat[:, i], label="Estimated", linestyle='--')
            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel(ylabels[i])
            ax[i].grid()
            ax[i].legend()
        fig.suptitle("Uncertainty estimation")
        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)

    def plot_tracking_performance(self, show=False):
        t = self.history('t')
        p_d = self.history('p_d')
        p = self.history('p')

        fig, ax = plt.subplots(3, 1, figsize=(8, 6))
        ylabels = ["x (m)", "y (m)", "z (m)"]
        for i in range(3):
            ax[i].plot(t, p_d[:, i], label="Desired")
            ax[i].plot(t, p[:, i], label="Actual")
            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel(ylabels[i])
            ax[i].grid()
            ax[i].legend()
        fig.suptitle("Trajectory")
        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)


def main():
    model = Model()
    simulator = Simulator(model)
    simulator.propagate(0.01, 60., True)
    # model.quadrotor_dyn.default_plot(show=False)
    model.plot_tracking_performance(show=False)
    model.plot_actuator_log(show=False)
    model.plot_uncertainty(show=True)


if __name__ == "__main__":
    main()
