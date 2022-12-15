import numpy as np

from pysimenv.core.base import DiscreteTimeDynSystem
from pysimenv.core.simulator import Simulator


class ConstAccel(DiscreteTimeDynSystem):
    def __init__(self, p_0: np.ndarray, v_0: np.ndarray, **kwargs):
        super(ConstAccel, self).__init__(
            initial_states={'p': p_0, 'v': v_0}, **kwargs
        )

    # implement
    def _update(self, p, v, a):
        T = self.interval

        p_next = p + T*v + 0.5*(T**2)*a
        v_next = v + T*a
        return {'p': p_next, 'v': v_next}


def main():
    p_0 = np.array([0., 0.])
    v_0 = np.array([10., 0.])

    model = ConstAccel(p_0, v_0, interval=0.5, name="model")
    simulator = Simulator(model)
    simulator.propagate(0.01, 10., save_history=True, a=np.array([-2., 1.]))
    model.default_plot(show=True)


if __name__ == "__main__":
    main()
