import numpy as np
import matplotlib.pyplot as plt
from pysimenv.core.base import DynSystem
from pysimenv.core.simulator import Simulator


class Projectile(DynSystem):
    def __init__(self, p_0, v_0, mu=0.):
        super(Projectile, self).__init__(initial_states={'p': p_0, 'v': v_0})
        self.mu = mu

    # implement
    def _deriv(self, p, v):
        p_dot = v.copy()
        v_dot = np.array([0., -9.807]) - self.mu*np.linalg.norm(v)*v
        return {'p': p_dot, 'v': v_dot}

    # implement
    def _check_stop_condition(self):
        p_y = self.state('p')[1]
        return p_y < 0.


def main():
    mu_list = {'no_drag': 0., 'air': 0.0214}
    data_list = dict()

    theta_0 = np.deg2rad(30.)
    p_0 = np.zeros(2)
    v_0 = np.array([20.*np.cos(theta_0), 20.*np.sin(theta_0)])
    for key, mu in mu_list.items():
        projectile = Projectile(p_0=p_0, v_0=v_0, mu=mu)
        simulator = Simulator(projectile)
        simulator.propagate(dt=0.01, time=10., save_history=True)

        data = projectile.history('t', 'p', 'v')
        data_list[key] = data

    color = {'no_drag': 'k', 'air': 'b'}
    fig, ax = plt.subplots(3, 1)
    for key in data_list.keys():
        data = data_list[key]
        ax[0].plot(data['p'][:, 0], data['p'][:, 1], color=color[key], label=key)
        ax[1].plot(data['t'], data['v'][:, 0], color=color[key], label=key)
        ax[2].plot(data['t'], data['v'][:, 1], color=color[key], label=key)

    ax[0].set_xlabel("Distance (m)")
    ax[0].set_ylabel("Height (m)")
    ax[0].set_title("Trajectory")
    ax[0].grid()
    ax[0].legend()

    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("v_x (m/s)")
    ax[1].set_title("Horizontal velocity")
    ax[1].grid()
    ax[1].legend()

    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("v_y (m/s)")
    ax[2].set_title("Vertical velocity")
    ax[2].grid()
    ax[2].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
