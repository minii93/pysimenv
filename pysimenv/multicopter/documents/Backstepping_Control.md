## Backstepping control

##### 2022/08/17

#### Dynamic model

The approximated dynamic model can be written as
$$
\begin{align}
\ddot{\phi} &= \frac{I_{y} - I_{z}}{I_{x}}\dot{\phi}\dot{\psi} + \frac{1}{I_{x}}\tau_{\phi} \\
\ddot{\theta} &= \frac{I_{z} - I_{x}}{I_{y}}\dot{\phi}\dot{\psi} + \frac{1}{I_{y}}\tau_{\theta} \\
\ddot{\psi} &= \frac{I_{x} - I_{y}}{I_{z}}\dot{\phi}\dot{\theta} + \frac{1}{I_{z}}\tau_{\psi} \\
\ddot{z} &= g - \frac{1}{m}c_{\phi}c_{\theta}f \\
\ddot{x} &= -\frac{1}{m}(c_{\phi}s_{\theta}c_{\psi} + s_{\phi}s_{\psi})f \\
\ddot{y} &= -\frac{1}{m}(c_{\phi}s_{\theta}c_{\psi} - s_{\phi}c_{\psi})f
\end{align}
$$
Define the state as $x_{1}=\phi, \, x_{2}=\dot{\phi}, \, x_{3}=\theta,\, x_{4}=\dot{\theta},\, x_{5}=\psi,\, x_{6}=\dot{\psi},\, x_{7}=z,\, x_{8}=\dot{z},\, x_{9}=x,\, x_{10}=\dot{x},\, x_{11}=y,\, x_{12}=\dot{y}$ and the control input as $u_{1}=f,\, u_{2}=\tau_{\phi},\, u_{3}=\tau_{\theta},\, u_{4}=\tau_{\psi}$.

The orientation subsystem can be expressed as
$$
\begin{align}
\dot{x}_{1} &= x_{2} \\
\dot{x}_{2} &= a_{1}x_{4}x_{6} + b_{1}u_{2} \\
\dot{x}_{3} &= x_{4} \\
\dot{x}_{4} &= a_{3}x_{2}x_{6} + b_{2}u_{3} \\
\dot{x}_{5} &= x_{6} \\
\dot{x}_{6} &= a_{5}x_{2}x_{4} + b_{3}u_{4}
\end{align}
$$
where
$$
a_{1}=\frac{I_{y} - I_{z}}{I_{x}}, a_{3} = \frac{I_{z} - I_{x}}{I_{y}}, a_{5}=\frac{I_{x} - I_{y}}{I_{z}}, b_{1}=\frac{1}{I_{x}}, b_{2}=\frac{1}{I_{y}}, b_{3}=\frac{1}{I_{z}}
$$
The control law is designed as
$$
\begin{align}
u_{2} &= \frac{1}{b_{1}}(-a_{1}x_{4}x_{6} + z_{1} + \alpha_{1}(z_{2} - \alpha_{1}z_{1}) + \alpha_{2}z_{2}) \\
u_{3} &= \frac{1}{b_{2}}(-a_{3}x_{2}x_{6} + z_{3} + \alpha_{3}(z_{4} - \alpha_{3}z_{3}) + \alpha_{4}z_{4}) \\
u_{4} &= \frac{1}{b_{3}}(-a_{5}x_{2}x_{4} + z_{5} + \alpha_{5}(z_{6} - \alpha_{5}z_{5}) + \alpha_{6}z_{6})
\end{align}
$$
where
$$
\begin{align}
z_{1} &= x_{1d} - x_{1},\, z_{2} = \dot{z}_{1} + \alpha_{1}z_{1} \\
z_{3} &= x_{3d} - x_{3},\, z_{4} = \dot{z}_{3} + \alpha_{3}z_{3} \\
z_{5} &= x_{5d} - x_{5},\, z_{6} = \dot{z}_{5} + \alpha_{5}z_{5}
\end{align}
$$
and each $\alpha_{i}$ is positive gain.

The translation subsystem can be expressed as
$$
\begin{align}
\dot{x}_{7} &= x_{8} \\
\dot{x}_{8} &= g - \frac{1}{m}(c_{1}c_{3})u_{1} \\
\dot{x}_{9} &= x_{10} \\
\dot{x}_{10} &= -\frac{1}{m}u_{x}u_{1} \\
\dot{x}_{11} &= x_{12} \\
\dot{x}_{12} &= -\frac{1}{m}u_{y}u_{1}
\end{align}
$$
where
$$
\begin{align}
u_{x} &= c_{1}s_{3}c_{5} + s_{1}s_{5} \\
u_{y} &= c_{1}s_{3}s_{5} - s_{1}c_{5}
\end{align}
$$
The control input for altitude control is designed as
$$
u_{1} = -\frac{m}{c_{1}c_{3}}(-g + z_{7} + \alpha_{7}(z_{8} - \alpha_{7}z_{7}) + \alpha_{8}z_{8})
$$
For $x$, $y$ control, desired values for $u_{x}$, $u_{y}$ are designed as
$$
\begin{align}
u_{x} &= -\frac{m}{u_{1}}(z_{9} + \alpha_{9}(z_{10} - \alpha_{9}z_{9}) + \alpha_{10}z_{10}) \\
u_{y} &= -\frac{m}{u_{1}}(z_{11} + \alpha_{11}(z_{12} - \alpha_{11}z_{11}) + \alpha_{12}z_{12})
\end{align}
$$
Desired values for $x_{1}, x_{3}$ can be calculated as
$$
x_{1} = \arcsin(s_{5}u_{x} - c_{5}u_{y}), x_{3} = \arcsin \left( \frac{c_{5}u_{x} + s_{5}u_{y}}{c_{1}} \right)
$$
