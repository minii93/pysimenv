## Fixed-Time Observer-Based Safety Control

##### 2022/07/13

Contents of this document is based on the following reference:

S. Zhou, K. Guo, X. Yu, L. Guo, and L. Xie, "Fixed-Time Observer Based Safety Control for a Quadrotor UAV"



#### Mathematical model

The rotation matrix from the body frame to the inertial frame is
$$
\begin{align}
R_{IB} &=\begin{bmatrix}
c_{\psi} & -s_{\psi} & 0 \\
s_{\psi} & c_{\psi} & 0 \\
0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
c_{\theta} & 0 & s_{\theta} \\
0 & 1 & 0 \\
-s_{\theta} & 0 & c_{\theta}
\end{bmatrix} \begin{bmatrix}
1 & 0 & 0 \\
0 & c_{\phi} & -s_{\phi} \\
0 & s_{\phi} & c_{\phi}
\end{bmatrix} \\
&= \begin{bmatrix}
c_{\psi}c_{\theta} & c_{\psi}s_{\theta}s_{\phi} - s_{\psi}c_{\phi} & c_{\psi}s_{\theta}c_{\phi} + s_{\psi}s_{\phi} \\
s_{\psi}c_{\theta} & s_{\psi}s_{\theta}s_{\phi} + c_{\psi}c_{\theta} & s_{\psi}s_{\theta}c_{\phi} - c_{\psi}s_{\phi} \\
-s_{\theta} & c_{\theta}s_{\psi} & c_{\theta}c_{\phi}
\end{bmatrix}
\end{align}
$$
The nonlinear dynamic model is described as
$$
\begin{align}
\dot{p} &= v \\
\dot{v} &= ge_{3} - \frac{1}{m}R_{IV}F_{m}e_{3} \\
\dot{\eta} &= R_{0}(\eta)\omega \\
\dot{\omega} &= J^{-1}(-\omega \times (J\omega) + M)
\end{align}
$$
where
$$
R_{0}(\eta) = \begin{bmatrix}
1 & s_{\theta}c_{\theta}^{-1}s_{\phi} & s_{\theta}c_{\theta}^{-1}c_{\phi} \\
0 & c_{\phi} & -s_{\phi} \\
0 & c_{\theta}^{-1}s_{\phi} & c_{\theta}^{-1}c_{\phi}
\end{bmatrix}
$$


#### Actuator model

Consider the following quadrotor-X configuration. The direction of the $z$ axis is chosen as the downward direction.

<img src="./QuadRotorX.svg" alt="QuadRotorX" width=200px /> 

Then,
$$
\begin{bmatrix}
F_{m} \\ M_{x} \\ M_{y} \\ M_{z}
\end{bmatrix} = \begin{bmatrix}
1 & 1 & 1 & 1 \\
-\frac{d_{\phi}}{2} & \frac{d_{\phi}}{2} & \frac{d_{\phi}}{2} & -\frac{d_{\phi}}{2} \\
\frac{d_{\theta}}{2} & -\frac{d_{\theta}}{2} & \frac{d_{\theta}}{2} & -\frac{d_{\theta}}{2} \\
c_{\tau_{f}} & c_{\tau_{f}} & -c_{\tau_{f}} & -c_{\tau_{f}}
\end{bmatrix} \begin{bmatrix}
f_{1} \\ f_{2} \\ f_{3} \\ f_{4}
\end{bmatrix};\quad u = R_{u}f_{s}
$$
where $f_{i}$ ($i=1, 2, 3, 4$) is the thrust produced by each rotor acting on the body, and $R_{u}$ is the mapping matrix. $d_{\phi}$ represents the roll motor-to-motor distance, $d_{\theta}$ represents the pitch motor-to-motor distance, and $c_{\tau_{f}}$ is a constant reflecting the relationship between the thrust force and the moment produced in the $z$ axis.



#### Fault model

The actuator fault model is defined to include both gain fault and bias fault.
$$
f_{s}^{\ast}=\Gamma_{u}f_{s} + \rho; \quad u^{\ast}=\Gamma_{u}u + R_{u}\rho
$$
where $f_{s}$ denotes the commanded thrust force, $f_{s}^{\ast}$ denotes the actual thrust force, $u$ denotes the commanded control input of the dynamic model, and $u^{\ast}$ denotes the actual control input of the dynamic model. $\Gamma_{u}=\mathrm{diag} (\alpha_1, \alpha_{2}, \alpha_{3}, \alpha_{4})$ with $\alpha_{i} \in (0, 1]$ is an unknown parameter representing the gain fault resulted from the changes of motor parameters. $\rho=[\rho_{1} \, \rho_{2} \, \rho_{3} \, \rho_{4}]^{T}$ is the bias fault caused by motor malfunction.



#### Fixed-time Fault Estimator

Consider the state vector $x=[v_{z}\, p\, q\, r]^{T}$ where $v_{z}$ represents the velocity in the vertical direction of the inertial frame, and $p, q, r$ represent components of the angular velocity.
$$
\begin{align}
\dot{v_{z}} &= g - \frac{1}{m}c_{\theta}c_{\phi}F_{m}^{\ast} \\
\dot{p} &= \frac{J_{y} - J_{z}}{J_{x}}qr + \frac{1}{J_{x}}M_{x}^{\ast} \\
\dot{q} &= \frac{J_{z} - J_{x}}{J_{y}}pr + \frac{1}{J_{y}}M_{y}^{\ast} \\
\dot{r} &= \frac{J_{x} - J_{y}}{J_{z}}pq + \frac{1}{J_{z}}M_{z}^{\ast}
\end{align}
$$
Then, the state space equation for $x$ can be written as
$$
\dot{x} = f(x, t) + Bu^{\ast}
$$
where
$$
f(x, t) = \begin{bmatrix}
g \\ \frac{J_{y} - J_{z}}{J_{x}}qr \\ \frac{J_{z} - J_{x}}{J_{y}}pr \\ \frac{J_{x} - J_{y}}{J_{z}}pq
\end{bmatrix},\, B=\mathrm{diag}(\frac{c_{\phi}c_{\theta}}{m}, J_{x}^{-1}, J_{y}^{-1}, J_{z}^{-1})
$$
The state space equation can be written as
$$
\dot{x} = f(x, t) + Bu + B(u^{\ast} - u) = f(x, t) + Bu + \Delta
$$
The uncertainty due to the actuator fault is defined as
$$
\Delta = B(u^{\ast} - u) = B(R_{u}(\Gamma_{u}f_{s} + \rho) - \Gamma_{u}f_{s}) = BR_{u}(\rho - (I - \Gamma_{u})f_{s})
$$
The fixed-time fault estimator is designed as
$$
\begin{align}
\dot{z}_{1} &= -k_{1}(\vert e_{d}\vert^{\alpha}\mathrm{sign}(e_{d}) + \vert e_{d}\vert^{\beta}\mathrm{sign}(e_{d})) + f(x, t) + z_{2} + Bu \\
\dot{z}_{2} &= -k_{2}(\vert e_{d} \vert^{2\alpha -1}\mathrm{sign}(e_{d}) + \vert e_{d}\vert^{2\beta - 1}\mathrm{sign}(e_{d}))
\end{align}
$$
where $z_{1}$ and $z_{2}$ are the estimation results of the state vector $x$ and actuator fault $\Delta$, respectively. $e_{d}=z_{1}-x$ is the residual between actual and estimated values of the state vector. Parameters are chosen as $\alpha \in (0.5, 1)$ and $\beta \in (1, 1.5)$. Positive observer gains $(k_{1}, k_{2})$ are assigned such that the matrix
$$
A = \begin{bmatrix}
-k_{1} & 1 \\
-k_{2} & 0
\end{bmatrix}
$$
is Hurwitz.



#### Integral Sliding Mode Control

Let $\hat{\Delta}$ denote the estimated value of $\Delta$. Define the baseline state $x_{b}$ as
$$
x_{b} = x_{d}(t_{0}) + \int_{t_{0}}^{t}(f(x, t) + Bu_{b})\,d\tau
$$
where $x_{d}$ is the desired state vector, and $u_{b}$ represents the control output of baseline controller. An integral sliding manifold is defined as
$$
s = N(x_{d} - x_{b})
$$
where $N=\mathrm{diag}(c_{1}, c_{2}, c_{3}, c_{4})$ is a positive coefficient matrix. The control law is designed as
$$
u_{f} = -(NB)^{-1}(N\hat{\Delta} + \epsilon_{1}\mathrm{sign}(s) + \epsilon_{2}s),\, u=u_{b} + u_{f}
$$
where $\epsilon_{1}$ and $\epsilon_{2}$ are positive coefficient diagonal matrices. In practice, the signum function $\mathrm{sign}(\cdot)$ is replaced with function
$$
\sigma(s) = \frac{2}{\pi}\tan^{-1}(\Vert s \Vert)s
$$
to attenuate the chattering effect.
