## Sliding mode control

##### 2022/10/19

Contents of this document is based on the following reference:

F. Chen, R. Jiang, K. Zhang, B. Jiang, and G. Tao, "Robust Backstepping Sliding-Mode Control and Observer-Based Fault Estimation for a Quadrotor UAV"



#### Attitude control

The attitude dynamics can be written as
$$
\dot{\omega} = J^{-1}(-\omega \times (J\omega) + \tau)
$$
where $\omega$ is the angular velocity. Let $\eta$ denotes the Euler angles. Then, the approximated dynamics is
$$
\ddot{\eta} = J^{-1}(-\dot{\eta} \times (J\dot{\eta}) + \tau)
$$
Define the sliding variable as
$$
s = \dot{e} + Ce
$$
where $e=\eta_{d} - \eta$.
$$
\begin{align}
\dot{s} &= \ddot{e} + C\dot{e} \\
&= \ddot{\eta}_{d} - J^{-1}(-\dot{\eta} \times (J\dot{\eta}) + \tau) + C\dot{e}
\end{align}
$$
The control law is designed as
$$
\tau = \dot{\eta} \times (J\dot{\eta}) + J(\ddot{\eta}_{d} + C\dot{e} + K\mathrm{sign}(s))
$$
Choose the Lyapunov candidate function as
$$
V = \frac{1}{2}s^{T}s
$$
Then,
$$
\dot{V} = s^{T}\dot{s} = s^{T}(-K\mathrm{sign}(s)) \le 0
$$
