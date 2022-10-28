import numpy as np
from pysimenv.core.base import SimObject
from pysimenv.multicopter.model import MulticopterDyn, QuadBase
from pysimenv.multicopter.control import FLVelControl
from pysimenv.core.simulator import Simulator


class FLVelTracking(SimObject):
    def __init__(self):
        super(FLVelTracking, self).__init__()

        m = 0.5
        J = np.diag([4.85, 4.85, 8.81])*1e-3

        pos = np.zeros(3)
        vel = np.zeros(3)
        R_iv = np.identity(3)
        omega = np.zeros(3)
        self.quadrotor = MulticopterDyn([pos, vel, R_iv, omega], m, J)

        k_p_att = np.array([1600., 1600., 1600])
        k_d_att = np.array([80., 80., 80])
        k_p_vel = np.array([5., 5., 5.])
        self.vel_control = FLVelControl(m, J, k_p_att, k_d_att, k_p_vel)

        self._add_sim_objs([self.quadrotor, self.vel_control])

    # implement
    def _forward(self, v_d: np.ndarray = np.zeros(3)):
        v = self.quadrotor.vel
        eta = self.quadrotor.euler_ang
        omega = self.quadrotor.ang_vel

        u = self.vel_control.forward(v, eta, omega, v_d)
        self.quadrotor.forward(u=u)


def main():
    model_params = {'m': 0.5, 'J': np.diag([4.85, 4.85, 8.81])*1e-3}
    k_p_att = np.array([1600., 1600., 1600])
    k_d_att = np.array([80., 80., 80])
    k_p_vel = np.array([5., 5., 5.])

    dyn = MulticopterDyn(p_0=np.zeros(3), v_0=np.zeros(3), R_0=np.identity(3), omega_0=np.zeros(3), **model_params,
                         name='dynamic model')
    control = FLVelControl(**model_params, k_p_att=k_p_att, k_d_att=k_d_att, k_p_vel=k_p_vel, name='controller')
    model = QuadBase(dyn=dyn, control=control, name='quad-rotor')

    simulator = Simulator(model)
    simulator.propagate(0.01, 10., True, v_d=np.array([0.5, 0., 1.]))
    model.dyn.default_plot(show=True)


if __name__ == "__main__":
    main()

