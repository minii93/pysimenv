import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Tuple, Optional
from pysimenv.core.base import SimObject, StaticObject, ArrayType
from pysimenv.core.system import DynObject, MultipleSystem, DynSystem
from pysimenv.common import orientation, util


class SignalGenerator(StaticObject):
    def __init__(self, shaping_fun, interval: Union[int, float] = -1):
        super(SignalGenerator, self).__init__(interval=interval)
        self.shaping_fun = shaping_fun

    # implementation
    def _forward(self) -> Union[np.ndarray, dict]:
        return self.shaping_fun(self.time)


class Scope(StaticObject):
    def __init__(self):
        super(Scope, self).__init__()

    # implementation
    def _forward(self, **kwargs):
        self._logger.append(t=self.time, **kwargs)

    def plot(self, show: bool = False):
        var_names = list(self._logger.keys())
        var_names.remove('t')

        t = self.history('t')
        for var_name in var_names:
            var = self.history(var_name)

            fig, ax = plt.subplots()
            for i in range(var.shape[1]):
                ax.plot(t, var[:, i], label=var_name + "_" + str(i))
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Value")
                ax.grid()
                ax.legend()
            fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.01)


class FeedbackControl(MultipleSystem):
    def __init__(self, system: DynObject, control: StaticObject):
        super(FeedbackControl, self).__init__()
        self.system = system
        self.control = control

        self.attach_sim_objects([system, control])

    # implement
    def _forward(self, **kwargs) -> Union[None, np.ndarray, dict]:
        x = self.system.state()
        if isinstance(x, np.ndarray):
            u_fb = self.control.forward(x=x)
        else:
            u_fb = self.control.forward(**x, **kwargs)

        if isinstance(u_fb, np.ndarray):
            out = self.system.forward(u=u_fb)
        else:
            out = self.system.forward(**u_fb)
        return out


class OFBControl(MultipleSystem):
    """
    Output feedback control
    """
    def __init__(self, system: DynObject, control: StaticObject):
        super(OFBControl, self).__init__()
        self.system = system
        self.control = control

        self.attach_sim_objects([system, control])

    # implement
    def _forward(self, **kwargs) -> Union[None, np.ndarray, dict]:
        y = self.system.output
        if isinstance(y, np.ndarray):
            u_fb = self.control.forward(y=y)
        else:
            u_fb = self.control.forward(**y, **kwargs)

        if isinstance(u_fb, np.ndarray):
            out = self.system.forward(u=u_fb)
        else:
            out = self.system.forward(**u_fb)
        return out


class Sequential(MultipleSystem):
    def __init__(self, obj_list: Union[List[SimObject], Tuple[SimObject]]):
        super(Sequential, self).__init__()
        assert len(obj_list) > 0, "(sequential) Invalid obj_list!"
        self.obj_list = list(obj_list)
        self.first_obj = obj_list[0]
        self.other_obj_list = obj_list[1:]

        self.attach_sim_objects(obj_list)

    # implement
    def _forward(self, **kwargs) -> Union[None, np.ndarray, dict]:
        self.first_obj.forward(**kwargs)
        out = self.first_obj.output
        for obj in self.other_obj_list:
            if out is None:
                out = obj.forward()
            elif isinstance(out, dict):
                out = obj.forward(**out)
            else:
                out = obj.forward(u=out)

        return out


class FlatEarthEnv(object):
    grav_accel = 9.805


class LinSys(DynSystem):
    def __init__(self, x_0: ArrayType, A: np.ndarray, B: Optional[np.ndarray] = None, C: Optional[np.ndarray] = None):
        super(LinSys, self).__init__(initial_states={'x': x_0})
        self.A = A
        self.B = B
        if C is None:
            C = np.eye(self.state_dim, dtype=np.float32)
        self.C = C

    # implement
    def _deriv(self, x: np.ndarray, u: Union[None, np.ndarray] = None):
        if self.B is None or u is None:
            x_dot = self.A.dot(x)
        else:
            x_dot = self.A.dot(x) + self.B.dot(u)

        return {'x': x_dot}

    # implement
    def _output(self) -> np.ndarray:
        return self.C.dot(self.state('x'))


class FirstOrderLinSys(LinSys):
    def __init__(self, x_0: ArrayType, tau: float):
        A = np.array([[-1./tau]])
        B = np.array([[1./tau]])

        super(FirstOrderLinSys, self).__init__(x_0, A, B)
        self.tau = tau

    # implement
    def _output(self) -> np.ndarray:
        return self.state('x')[0]


class SecondOrderLinSys(LinSys):
    def __init__(self, x_0: ArrayType, zeta: float, omega: float):
        A = np.array([
            [0., 1.],
            [-omega**2, -2*zeta*omega]
        ], dtype=np.float32)
        B = np.array([[0.], [omega**2]], dtype=np.float32)

        super(SecondOrderLinSys, self).__init__(x_0, A, B)
        self.zeta = zeta
        self.omega = omega

    # implement
    def _output(self) -> np.ndarray:
        return self.state('x')[0]


class Integrator(DynSystem):
    def __init__(self, initial_state: Union[ArrayType]):
        def deriv_fun(x: np.ndarray, u: Union[float, np.ndarray]):
            if isinstance(u, float):
                u = np.array([u])
            return {'x': u}

        super(Integrator, self).__init__({'x': initial_state}, deriv_fun)


class Differentiator(StaticObject):
    def __init__(self):
        super(Differentiator, self).__init__()
        self.u_prev = None
        self.t_prev = None

    # implement
    def _forward(self, u: np.ndarray) -> np.ndarray:
        if self.u_prev is None:
            deriv = np.zeros_like(u)
        else:
            deriv = (u - self.u_prev)/(self.time - self.t_prev)

        if self._sim_clock.major_time_step:
            self.u_prev = u.copy()
            self.t_prev = self.time

        return deriv


class PIDControl(DynObject):
    def __init__(self, k_p: Union[float, np.ndarray], k_i: Union[float, np.ndarray], k_d: Union[float, np.ndarray],
                 windup_limit=None,
                 interval: Union[int, float] = -1):
        super(PIDControl, self).__init__(initial_states={'e_i': np.zeros_like(k_p)}, interval=interval)
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.windup_limit = windup_limit
        if self.windup_limit:
            self.state_vars['e_i'].attach_correction_fun(self.clip_windup)

        self.e_prev = None
        self.t_prev = None

    # implement
    def _forward(self, e: np.ndarray) -> np.ndarray:
        self.state_vars['e_i'].set_deriv(deriv=e)

        e_i = self.state('e_i')
        e_d = np.zeros_like(e) if self.e_prev is None else (e - self.e_prev)/(self.time - self.t_prev)
        u_pid = self.k_p*e + self.k_i*e_i + self.k_d*e_d

        if self._sim_clock.major_time_step:
            self.e_prev = e.copy()
            self.t_prev = self.time

        return u_pid

    def clip_windup(self, e_i: np.ndarray) -> np.ndarray:
        return np.clip(e_i, self.windup_limit, -self.windup_limit)


class Dyn6DOF(DynSystem):
    def __init__(self, p_0: np.ndarray, v_b_0: np.ndarray, q_0: np.ndarray, omega_0: np.ndarray,
                 m: float, J: np.ndarray):
        super(Dyn6DOF, self).__init__(initial_states={'p': p_0, 'v_b': v_b_0, 'q': q_0, 'omega': omega_0})
        self.m = m
        self.J = J

        def normalize(v: np.ndarray) -> np.ndarray:
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6:
                return np.zeros_like(v)
            else:
                return v/v_norm

        self.state_vars['q'].attach_correction_fun(normalize)

    @property
    def R(self) -> np.ndarray:
        # rotation matrix representing the rotation from the inertial frame to the body frame
        q = self.state('q')
        eta, epsilon = q[0], q[1:4]
        S_epsilon = orientation.hat(epsilon)
        R_bi = np.identity(3) - 2*eta*S_epsilon + 2*np.matmul(S_epsilon, S_epsilon)
        return R_bi

    @property
    def v_i(self) -> np.ndarray:
        v_b = self.state('v_b')
        R_ib = self.R.transpose()
        return R_ib.dot(v_b)

    # implement
    def _deriv(self, p, v_b, q, omega, f_b: np.ndarray, m_b: np.ndarray):
        p_dot = self.v_i
        v_b_dot = -np.cross(omega, v_b) + 1./self.m*f_b

        Q = np.array([
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]]
        ])
        q_dot = 0.5*Q.dot(omega)
        omega_dot = np.linalg.solve(self.J, -np.cross(omega, self.J.dot(omega)) + m_b)
        return {'p': p_dot, 'v_b': v_b_dot, 'q': q_dot, 'omega': omega_dot}

    # override
    def _forward(self, **kwargs):
        output = super(Dyn6DOF, self)._forward(**kwargs)
        self._logger.append(R=self.R, v_i=self.v_i)
        return output


class Dyn6DOFEuler(DynSystem):
    def __init__(self, p_0: np.ndarray, v_b_0: np.ndarray, eta_0: np.ndarray, omega_0: np.ndarray,
                 m: float, J: np.ndarray):
        super(Dyn6DOFEuler, self).__init__(initial_states={'p': p_0, 'v_b': v_b_0, 'eta': eta_0, 'omega': omega_0})
        self.m = m
        self.J = J
        self.name = "dyn_6dof_euler"

        self.state_vars['eta'].attach_correction_fun(util.wrap_to_pi)

    @property
    def R(self) -> np.ndarray:
        eta = self.state('eta')
        R_bi = orientation.euler_angles_to_rotation(eta)
        return R_bi

    @property
    def v_i(self) -> np.ndarray:
        v_b = self.state('v_b')
        R_ib = self.R.transpose()
        return R_ib.dot(v_b)

    # implement
    def _deriv(self, p, v_b, eta, omega, f_b: np.ndarray, m_b: np.ndarray):
        p_dot = self.v_i
        v_b_dot = -np.cross(omega, v_b) + 1./self.m*f_b

        phi, theta = eta[0:2]
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        t_theta = np.tan(theta)
        assert np.abs(theta) < np.pi/2. - 1e-6, "[{:s}] The attitude is near the singularity.".format(self.name)
        H_inv = np.array([
            [1., s_phi*t_theta, c_phi*t_theta],
            [0., c_phi, -s_phi],
            [0., s_phi/c_theta, c_phi/c_theta]
        ])
        eta_dot = H_inv.dot(omega)
        omega_dot = np.linalg.solve(self.J, -np.cross(omega, self.J.dot(omega)) + m_b)
        return {'p': p_dot, 'v_b': v_b_dot, 'eta': eta_dot, 'omega': omega_dot}

    # override
    def _forward(self, **kwargs):
        output = super(Dyn6DOFEuler, self)._forward(**kwargs)
        self._logger.append(R=self.R, v_i=self.v_i)
        return output


class Dyn6DOFRotMat(DynSystem):
    def __init__(self, p_0: np.ndarray, v_b_0: np.ndarray, R_0: np.ndarray, omega_0: np.ndarray,
                 m: float, J: np.ndarray):
        super(Dyn6DOFRotMat, self).__init__(initial_states={'p': p_0, 'v_b': v_b_0, 'R': R_0, 'omega': omega_0})
        self.m = m
        self.J = J
        self.name = "dyn_6dof_rotation"

        self.state_vars['R'].attach_correction_fun(orientation.correct_orthogonality)

    @property
    def v_i(self) -> np.ndarray:
        v_b = self.state('v_b')
        R_ib = self.state('R').transpose()
        return R_ib.dot(v_b)

    # implement
    def _deriv(self, p, v_b, R, omega, f_b: np.ndarray, m_b: np.ndarray):
        p_dot = self.v_i
        v_b_dot = -np.cross(omega, v_b) + 1./self.m*f_b
        R_dot = -np.matmul(orientation.hat(omega), R)
        omega_dot = np.linalg.solve(self.J, -np.cross(omega, self.J.dot(omega)) + m_b)
        return {'p': p_dot, 'v_b': v_b_dot, 'R': R_dot, 'omega': omega_dot}

    # override
    def _forward(self, **kwargs):
        output = super(Dyn6DOFRotMat, self)._forward(**kwargs)
        self._logger.append(v_i=self.v_i)
        return output

