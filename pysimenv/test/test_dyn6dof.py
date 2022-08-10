import numpy as np
from pysimenv.common.model import Dyn6DOF, SignalGenerator, Sequential
from pysimenv.core.simulator import Simulator


def main():
    m = 1.
    w, h, d = 0.3, 0.1, 0.5
    J = np.diag([1./12*m*(h**2 + d**2), 1./12*m*(w**2 + h**2), 1./12*m*(w**2 + d**2)])

    dyn6dof = Dyn6DOF(
        p_0=np.zeros(3), v_b_0=np.zeros(3),
        q_0=np.array([1., 0., 0., 0.]), omega_0=np.zeros(3),
        m=m, J=J)

    def force_moment(t):
        if t < 2.:
            f_b = np.array([0., 0., 1.])
            m_b = np.array([0.15, 0., 0.])
        else:
            f_b = np.zeros(3)
            m_b = np.zeros(3)
        return {'f_b': f_b, 'm_b': m_b}
    source = SignalGenerator(shaping_fun=force_moment)
    model = Sequential(obj_list=[source, dyn6dof])
    simulator = Simulator(model)
    simulator.propagate(dt=0.01, time=5., save_history=True)
    dyn6dof.default_plot(show=True)


if __name__ == "__main__":
    main()
