import numpy as np
import ray
import matplotlib.pyplot as plt
from pysimenv.core.simulator import Simulator, ParallelSimulator
from pysimenv.common.model import SecondOrderLinSys


def model_generator(zeta, omega):
    model = SecondOrderLinSys(
        initial_state=[0, 0], zeta=zeta, omega=omega)
    return model


@ray.remote
def simulation_fun(zeta, omega):
    model = model_generator(zeta, omega)
    u_step = 1.
    simulator = Simulator(model, verbose=False)
    simulator.propagate(0.01, 100, True, u_step)

    state = model.history('x')
    overshoot = max(state[:, 0] - u_step)/u_step*100.

    return {'overshoot': overshoot}


def main():
    zeta_array = np.linspace(0.2, 1., 5)
    omega_array = np.linspace(1., 10., 10)

    parameter_sets = []
    for zeta in zeta_array:
        for omega in omega_array:
            parameter_sets.append(
                {'zeta': zeta, 'omega': omega}
            )

    par_simulator = ParallelSimulator(simulation_fun)
    par_simulator.simulate(parameter_sets=parameter_sets, verbose=True)
    par_simulator.save('./data/')

    data = par_simulator.get('zeta', 'omega', 'overshoot')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(
        data['zeta'],
        data['omega'],
        data['overshoot'], c=data['overshoot'])
    ax.set_xlabel('Zeta')
    ax.set_ylabel('Omega')
    ax.set_zlabel('Overshoot (%)')
    ax.grid()
    fig.colorbar(sc)
    plt.show()


if __name__ == "__main__":
    main()
