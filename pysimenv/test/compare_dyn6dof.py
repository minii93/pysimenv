import numpy as np
import matplotlib.pyplot as plt
from pysimenv.common.model import Dyn6DOF, Dyn6DOFEuler, Dyn6DOFRotMat, SignalGenerator, Sequential
from pysimenv.core.simulator import Simulator


def main():
    m = 1.
    w, h, d = 0.3, 0.1, 0.5
    J = np.diag([1./12*m*(h**2 + d**2), 1./12*m*(w**2 + h**2), 1./12*m*(w**2 + d**2)])

    p_0 = np.zeros(3)
    v_b_0 = np.zeros(3)
    omega_0 = np.zeros(3)

    dyn6dof = Dyn6DOF(
        p_0=p_0, v_b_0=v_b_0, q_0=np.array([1., 0., 0., 0.]), omega_0=omega_0,
        m=m, J=J)
    dyn6dof_euler = Dyn6DOFEuler(
        p_0=p_0, v_b_0=v_b_0, eta_0=np.zeros(3), omega_0=omega_0, m=m, J=J
    )
    dyn6dof_rot_mat = Dyn6DOFRotMat(
        p_0=p_0, v_b_0=v_b_0, R_0=np.identity(3), omega_0=omega_0, m=m, J=J
    )

    def force_moment(t):
        f_b = np.zeros(3)
        if t < 2.5:
            m_b = np.array([0.01, 0., 0.])
        elif t < 5.:
            m_b = np.array([0., 0., 0.01])
        else:
            m_b = np.zeros(3)
        return {'f_b': f_b, 'm_b': m_b}
    source = SignalGenerator(shaping_fun=force_moment)
    models = {
        'quaternion': Sequential(obj_list=[source, dyn6dof]),
        'euler': Sequential(obj_list=[source, dyn6dof_euler]),
        'rot_mat': Sequential(obj_list=[source, dyn6dof_rot_mat])
    }
    line_styles = {'quaternion': '-', 'euler': '-.', 'rot_mat': ':'}
    rotations = dict()
    for key, model in models.items():
        simulator = Simulator(model)
        simulator.propagate(dt=0.01, time=20., save_history=True)
        rotations[key] = {'t': model.obj_list[1].history('t'), 'R': model.obj_list[1].history('R')}

    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            for key, data in rotations.items():
                t = data['t']
                R = data['R']
                ax[i, j].plot(t, R[:, i, j], label=key, linestyle=line_styles[key])
                ax[i, j].set_xlabel("Time (s)")
                ax[i, j].set_ylabel("R({:d},{:d})".format(i, j))
                ax[i, j].grid()
                ax[i, j].legend()
    plt.show()


if __name__ == "__main__":
    main()
