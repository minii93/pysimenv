## FTC in Total Failure of One Actuator

##### 2022/08/12

#### Dynamics model

$p=[x \, y \, z]^{T}$ is the position in the inertial frame, $v=[v_{x} \, v_{y}\, v_{z}]^{T}$ is the velocity in the inertial frame, $\eta=[\phi \, \theta \, \psi]$ is the Euler angles, and $\omega=[p \, q \, r]^{T}$ is the angular velocity expressed in the body frame.
$$
\begin{align}
\dot{p} &= v \\
\dot{v} &= \frac{1}{m}(mge_{3} - RF_{l}e_{3} - d_{v}v) \\
\dot{\eta} &= H(\eta)^{-1}\omega \\
\dot{\omega} &= J^{-1}(-\omega^{\times}J\omega - d_{\omega}\omega + \tau)
\end{align}
$$
where $d_{v}=\mathrm{diag}(d_{x}, d_{y}, d_{z})$ is the lumped drag force coefficient, $d_{\omega}=\mathrm{diag}(d_{\phi}, d_{\theta}, d_{\psi})$ is the lumped drag torque coefficient, and
$$
H(\eta)=\begin{bmatrix}
1 & 0 & -s_{\theta} \\
0 & c_{\phi} & s_{\phi}c_{\theta} \\
0 & -s_{\phi} & c_{\phi}c_{\theta}
\end{bmatrix},\quad 
H(\eta)^{-1} = \begin{bmatrix}
1 & s_{\phi}t_{\theta} & c_{\phi}t_{\theta} \\
0 & c_{\phi} & -s_{\phi} \\
0 & s_{\phi}/c_{\theta} & c_{\phi}/c_{\theta}
\end{bmatrix}
$$

The dynamic equations can be expressed as
$$
\begin{align}
\dot{x} &= v_{x}, \dot{y} = v_{y}, \dot{z}=v_{z} \\
\dot{v}_{x} &= \frac{1}{m}(-(c_{\phi}s_{\theta}c_{\psi} + s_{\phi}s_{\psi})F_{l} - d_{x}v_{x}) \\
\dot{v}_{y} &= \frac{1}{m}(-(c_{\phi}s_{\theta}s_{\psi} - s_{\phi}c_{\psi})F_{l} - d_{y}v_{y}) \\
\dot{v}_{z} &= g + \frac{1}{m}(-(c_{\phi}c_{\theta})F_{l} - d_{z}v_{z}) \\
\dot{\phi} &= p + s_{\phi}t_{\theta}q + c_{\phi}t_{\theta}r \\
\dot{\theta} &= c_{\phi}q - s_{\phi}r \\
\dot{\psi} &= \frac{s_{\phi}}{c_{\theta}}q + \frac{c_{\phi}}{c_{\theta}}r \\
\dot{p} &= \frac{1}{I_{xx}}((I_{yy} - I_{zz})qr - d_{\phi}p + \tau_{\phi}) \\
\dot{q} &= \frac{1}{I_{yy}}((I_{zz} - I_{xx})pr - d_{\theta}q + \tau_{\theta}) \\
\dot{r} &= \frac{1}{I_{zz}}((I_{xx} - I_{yy})pq - d_{\psi}r + g_{\psi}u_{r})
\end{align}
$$



#### Altitude control

According to the dynamic equations,
$$
\ddot{z} = g - \frac{1}{m}c_{\phi}c_{\theta}F_{l} - \frac{1}{m}d_{z}v_{z}
$$
Define $e_{1}=z - z_{d}$, $e_{2}=\dot{e}_{1}+ k_{1}e_{1}$. Using back-stepping control, the control law for the altitude control is designed as
$$
\begin{align}
F_{l} &= -\frac{m}{c_{\phi}c_{\theta}}\left( -g + \frac{1}{m}d_{z}v_{z} + (1 - k_{1}^{2})e_{1} + (k_{1} + k_{2})e_{2} + \ddot{z}_{d} \right) \\
&= -\frac{m}{c_{\phi}c_{\theta}}\left( -g + \frac{1}{m}d_{z}v_{z} + (1 + k_{1}k_{2})e_{1} + (k_{1} + k_{2})\dot{e}_{1} + \ddot{z}_{d} \right)
\end{align}
$$


#### Reduced model for attitude control

The control inputs $u=[F_{l}, \tau_{\phi} \, \tau_{\theta} \, \tau_{\psi}]^{T}$ can be calculated using thrust of each motor $w=[F_{1} \, F_{2} \, F_{3} \, F_{4}]^{T}$ as
$$
u = Gw
$$
where $G$ is an invertible $4 \times 4$ matrix called the effective matrix, which is determined according to the configuration of the quadrotor. Assume that the $k$th motor has experienced the total failure. Define $\mathcal{I} = \{1, 2, 3\}$, $\mathcal{J}=\{j \vert 1 \le j \le 4, j \ne k \}$, $u_{r} = [F_{l}, \tau_{\phi} \, \tau_{\theta}]^{T}$ and $w_{r}=w_{\mathcal{J}}$. For example, if the second motor has experienced the failure, then $\mathcal{J}=\{1, 3, 4\}$ and $w_{r}=[F_{1}\, F_{3} \, F_{4}]^{T}$.
$$
u_{r} = G_{\mathcal{I}, \mathcal{J}}w_r; \quad w_{r} = G_{\mathcal{I}, \mathcal{J}}^{-1}u_{r}; \quad \therefore \tau_{\psi}=G_{4, \mathcal{J}}w_{r} = G_{4, \mathcal{J}}G_{\mathcal{I}, \mathcal{J}}^{-1}u_{r}
$$
 Defining $g_{\psi}=G_{4, \mathcal{J}}G_{\mathcal{I}, \mathcal{J}}^{-1}$, we get $\tau_{\psi} = g_{\psi}u_{r}$.

The time derivative for the angular velocity can be expressed as
$$
\begin{align}
\dot{\omega} &= J^{-1}(-\omega^{\times}J\omega - d_{\omega}\omega) + J^{-1}\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
g_{\psi, 1} & g_{\psi, 2} & g_{\psi, 3}
\end{bmatrix}u_{r}, \quad u_{r}=\begin{bmatrix} F_{l} \\ \tau_{r} \end{bmatrix}
\end{align}
$$
where $\tau_{r}=[\tau_{\phi} \, \tau_{\theta}]^{T}$.



#### Attitude control

Let $\zeta=[\phi \, \theta]^{T}$. Then,
$$
\dot{\zeta} = \begin{bmatrix} \dot{\phi} \\ \dot{\theta} \end{bmatrix}
= \begin{bmatrix}
1 & s_{\phi}t_{\theta} & c_{\phi}t_{\theta} \\
0 & c_{\phi} & -s_{\phi}
\end{bmatrix} \begin{bmatrix}
p \\ q \\ r
\end{bmatrix} =: M_{1}\omega
$$
Differentiating with respect to time,
$$
\ddot{\zeta} = \dot{M}_{1}\omega + M_{1}\dot{\omega}
$$
where
$$
\begin{align}
\dot{\omega} &= J^{-1}(-\omega^{\times}J\omega - d_{\omega}\omega) + J^{-1}\begin{bmatrix}
0 \\ 0 \\ g_{\psi, 1}
\end{bmatrix}F_{l} + J^{-1} \begin{bmatrix}
1 & 0 \\
0 & 1 \\
g_{\psi, 2} & g_{\psi, 3}
\end{bmatrix}\tau_{r} \\
&=: n_{1} + M_{2}\tau_{r}
\end{align}
$$
and
$$
\dot{M}_{1} = \begin{bmatrix}
0 & c_{\phi}t_{\theta}\dot{\phi} + s_{\phi}/c_{\theta}^{2}\dot{\theta} & -s_{\phi}t_{\theta}\dot{\phi} + c_{\phi}/c_{\theta}^{2}\dot{\theta} \\
0 & -s_{\phi}\dot{\phi} & -c_{\phi}\dot{\phi}
\end{bmatrix}
$$
Therefore,
$$
\begin{align}
\ddot{\zeta} &= \dot{M}_{1}\omega + M_{1}(n_{1} + M_{2}u_{r}) = \dot{M}_{1}\omega + M_{1}n_{1} + M_{1}M_{2}\tau_{r} \\
&=: f + Q\tau_{r}
\end{align}
$$
Design the control law as $\tau_{r}=Q^{\dagger}(-f + v)$ where $Q^{\dagger}$ denotes the Moore-Penrose inverse of $Q$.
$$
\begin{align}
\ddot{\zeta} &= f + QQ^{\dagger}(-f + v) \\
&= (I - QQ^{\dagger})f + QQ^{\dagger}v
\end{align}
$$
When $Q$ is nonsingular, $QQ^{\dagger} = I$, and $\ddot{\zeta} = v$.

Using back-stepping control, the control law for the attitude control can be designed as
$$
\tau_{r} = Q^{\dagger}(-f + (I - K_{3}^{2})e_{3} + (K_{3} + K_{4})e_{4} + \ddot{\zeta}_{d})
$$
where $e_{3}=\zeta_{d} - \zeta$, $e_{4}=\dot{e}_{3} + K_{3}e_{3}$.
$$
\therefore \tau_{r} = Q^{\dagger}(-f + (I + K_{3}K_{4})e_{3} + (K_{3} + K_{4})\dot{e}_{3} + \ddot{\zeta}_{d})
$$





#### Appendix (Backstepping control)

$v$ is designed using backstepping control. Consider the system
$$
\begin{align}
\dot{x}_{1} &= x_{2} \\
\dot{x}_{2} &= u
\end{align}
$$
with the virtual system
$$
\dot{x}_{1}=v
$$
(Step 1) Let the first tracking error as $e_{1}=x_{1d}-x_{1}$. The Lyapunov candidate function $V_{1}$ is defined as
$$
V_{1}=\frac{1}{2}e_{1}^{T}e_{1}
$$
Its time derivative is
$$
\dot{V}_{1} = \dot{e}_{1}^{T}e_{1} = (\dot{x}_{1d} - v)^{T}e_{1}
$$
Design
$$
\dot{x}_{1d}-v = -K_{1}e_{1}\quad \therefore v = \dot{x}_{1d} + K_{1}e_{1}
$$
(Step 2) Let the second tracking error as $e_{2}=v-x_{2}$. Then,
$$
e_{2} = \dot{x}_{1d} + K_{1}e_{1} - \dot{x}_{1} = \dot{e}_{1} + K_{1}e_{1}
$$
The Lyapunov candidate function $V_{2}$ is defined as
$$
V_{2}=\frac{1}{2}e_{1}^{T}e_{1} + \frac{1}{2}e_{2}^{T}e_{2}
$$
Its time derivative is
$$
\begin{align}
\dot{V}_{2} &= \dot{e}_{1}^{T}e_{1} + \dot{e}_{2}^{T}e_{2} \\
&= (e_{2} - K_{1}e_{1})^{T}e_{1} + (\dot{v} - \dot{x}_{2})^{T}e_{2} \\
&= -e_{1}^{T}K_{1}e_{1} + e_{2}^{T}(e_{1} + \dot{v} - \dot{x}_{2})
\end{align}
$$
Design
$$
\begin{align}
e_{1} + \dot{v} - \dot{x}_{2} &= -K_{2}e_{2} \\
\therefore u &= e_{1} + K_{2}e_{2} + \dot{v} \\
&= e_{1} + K_{2}e_{2} + \dot{x}_{2d} + K_{1}\dot{e}_{1} \\
&= e_{1} + K_{2}e_{2} + \dot{x}_{2d} + K_{1}(e_{2} - K_{1}e_{1}) \\
&= (I - K_{1}^{2})e_{1} + (K_{1} + K_{2})e_{2} + \dot{x}_{2d}
\end{align}
$$




