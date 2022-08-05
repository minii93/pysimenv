import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.system import DynObject, DynSystem, MultipleSystem
from pysimenv.core.simulator import Simulator


class PIController(DynObject):
    def __init__(self, k_p, k_i):
        super(PIController, self).__init__(initial_states={'e_i': np.array([0.])})
        self.k_p = k_p
        self.k_i = k_i

    # implement
    def _forward(self, e):
        self.state_vars['e_i'].set_deriv(deriv=e)

        e_i = self.state['e_i']
        u_pi = self.k_p*e + self.k_i*e_i
        return u_pi


class CCCar(MultipleSystem):
    """
    Cruise-controlled car
    """
    def __init__(self, k_p, k_i, v_0):
        super(CCCar, self).__init__()
        # mass-normalized parameters
        c = 0.02
        g = 9.81

        def deriv_fun(v, u, theta):
            v_dot = -c*v + u - g*theta
            return {'v': v_dot}

        self.vel_dyn = DynSystem(
            initial_states={'v': v_0},
            deriv_fun=deriv_fun
        )

        # PI Controller
        self.pi_control = PIController(k_p=k_p, k_i=k_i)

        self.attach_sim_objects([self.vel_dyn, self.pi_control])

    # implement
    def _forward(self, v_r, theta):
        # tracking error
        v = self.vel_dyn.state['v']
        e = v_r - v

        # PI control input
        u_pi = self.pi_control.forward(e=e)
        u = np.clip(u_pi, 0., 1.)

        # update dynamic
        self.vel_dyn.forward(u=u, theta=theta)

        # log velocity error and control signal
        self._logger.append(t=self.time, e=e, u=u)


def main():
    zeta = 1.
    omega_0_list = [0.05, 0.1, 0.2]
    v_r = 5.
    theta = 0.04  # 4% slope

    data_list = []

    for omega_0 in omega_0_list:
        k_p = 2*zeta*omega_0 - 0.02
        k_i = omega_0**2

        car = CCCar(k_p=k_p, k_i=k_i, v_0=v_r)
        simulator = Simulator(car)
        simulator.propagate(dt=0.01, time=100, save_history=True, v_r=v_r, theta=theta)

        data = car.history('t', 'e', 'u')  # returns dictionary when the number of variable is greater than 1
        data_list.append(data)

    fig, ax = plt.subplots(2, 1)
    lines = [':', '-', '--']
    for i in range(3):
        data = data_list[i]
        ax[0].plot(data['t'], data['e'], color='b', linestyle=lines[i])
        ax[1].plot(data['t'], data['u'], color='b', linestyle=lines[i])

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Velocity error")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Control signal")
    ax[0].set_xticks(np.linspace(0., 100., 11))
    ax[0].set_yticks(np.linspace(0., 4., 5))
    ax[1].set_xticks(np.linspace(0., 100., 11))
    ax[1].set_yticks(np.linspace(0., 0.6, 7))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

