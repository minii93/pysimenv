import numpy as np
from typing import Union, List
from pysimenv.core.base import SimObject
from pysimenv.core.simulator import Simulator
from pysimenv.multicopter.base import MulticopterDyn, EffectorModel, Mixer


class QuadBase(SimObject):
    def __init__(self, dyn: MulticopterDyn, control: SimObject, effector: EffectorModel = None, mixer: Mixer = None, **kwargs):
        super(QuadBase, self).__init__(**kwargs)
        self.dyn = dyn
        self.control = control
        self.effector = effector
        self.mixer = mixer
        self._add_sim_objs([self.dyn, self.control])

    # implementation
    def _forward(self, **kwargs):
        """
        :param sigma_d: [x_d, y_d, z_d, phi_d]
        :param sigma_d_dot: [x_d_dot, y_d_dot, z_d_dot, phi_d_dot]
        :return:
        """
        if self.effector:
            if self.mixer:
                u_cmd = self.control.forward(self.dyn, **kwargs)  # u_cmd = {'f': total force, 'tau': moments}
                thrusts = self.mixer.convert(**u_cmd)
            else:
                thrusts = self.control.forward(self.dyn, **kwargs)  # thrusts = [f_1, f_2, ..., f_m]
            u = self.effector.convert(thrusts)
        else:
            u = self.control.forward(self.dyn, **kwargs)

        self.dyn.forward(**u)

    # implementation
    def _output(self) -> dict:
        return self.dyn.output


class QuadXEffector(EffectorModel):
    def __init__(self, d_phi: float, d_theta: float, c_tau_f: float):
        self.d_phi = d_phi
        self.d_theta = d_theta
        self.c_tau_f = c_tau_f
        self.B = np.array([
            [1., 1., 1., 1.],
            [-d_phi/2., d_phi/2., d_phi/2., -d_phi/2.],
            [d_theta/2., -d_theta/2., d_theta/2., -d_theta/2.],
            [c_tau_f, c_tau_f, -c_tau_f, -c_tau_f]
        ])  # mapping matrix

    def convert(self, f: np.ndarray) -> dict:
        u = self.B.dot(f)
        return {'f': u[0], 'tau': u[1:4]}


class QuadXMixer(Mixer):
    def __init__(self, d_phi: float, d_theta: float, c_tau_f: float):
        self.d_phi = d_phi
        self.d_theta = d_theta
        self.c_tau_f = c_tau_f
        self.B = np.array([
            [1., 1., 1., 1.],
            [-d_phi/2., d_phi/2., d_phi/2., -d_phi/2.],
            [d_theta/2., -d_theta/2., d_theta/2., -d_theta/2.],
            [c_tau_f, c_tau_f, -c_tau_f, -c_tau_f]
        ])  # mapping matrix
        self.B_inv = np.linalg.inv(self.B)

    def convert(self, f: float, tau: np.ndarray) -> np.ndarray:
        f = self.B_inv.dot(np.array([f, tau[0], tau[1], tau[2]]))
        return f


class ActuatorFault(SimObject):
    def __init__(self, t_list: List[float], alp_list: List[np.ndarray], rho_list: List[np.ndarray],
                 interval: Union[int, float] = -1):
        """
        :param t_list: [t_0, t_1, ..., t_{k-1}] time of fault occurrence
        :param alp_list: [lam_0: (M,) array, ..., lam_{k-1}] gain fault, M: the number of motors
        :param rho_list: [rho_0: (M,) array, ..., rho_{k-1}] bias fault, M: the number of motors
        :return:
        """
        super(ActuatorFault, self).__init__(interval)
        assert len(t_list) == len(alp_list), "Array sizes doesn't match."
        assert len(t_list) == len(rho_list), "Array sizes doesn't match."
        self.t_list = t_list
        self.alp_list = alp_list
        self.rho_list = rho_list

        self.alp = alp_list[0]
        self.rho = rho_list[0]
        self.next_ind = 1

    # implement
    def _forward(self, f_s: np.ndarray):
        if self.next_ind < len(self.t_list):
            if self.time >= self.t_list[self.next_ind]:
                self.alp = self.alp_list[self.next_ind]
                self.rho = self.rho_list[self.next_ind]
                self.next_ind += 1
        f_s_star = self.alp*f_s + self.rho
        return f_s_star


def main():
    model_params = {'m': 4.34, 'J': np.diag([0.0820, 0.0845, 0.1377])}

    quad_dyn = MulticopterDyn(p_0=np.array([0., 0., -1.]), v_0=np.array([1., 0., 0.]),
                              R_0=np.identity(3), omega_0=np.array([0., 0., 0.1]), **model_params,
                              name="dynamic model")
    f = 45.
    tau = np.zeros(3)

    simulator = Simulator(quad_dyn)
    simulator.propagate(dt=0.01, time=10., save_history=True, f=f, tau=tau)

    quad_dyn.default_plot()
    quad_dyn.plot_euler_angles(show=True)


if __name__ == "__main__":
    main()

