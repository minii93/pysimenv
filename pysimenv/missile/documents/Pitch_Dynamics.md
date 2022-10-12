## Pitch dynamics

##### 2022/09/07

The linearized aerodynamic model is given by
$$
\begin{align}
\frac{L}{m} &= L_{\alpha}\alpha + L_{\delta}\delta \\
\frac{M}{I} &= M_{\alpha}\alpha + M_{q}q + M_{\delta}\delta
\end{align}
$$
where $L$ is the lift force, $M$ is the aerodynamic moment, $m$ is the mass, and $I$ is the moment of inertia.

The missile dynamics in the pitch plane are given by
$$
\begin{align}
\dot{\alpha} &= q - \frac{L(\alpha, \delta)}{mV_{M}} = q - \frac{L_{\alpha}\alpha + L_{\delta}\delta}{V_{M}} \\
\dot{\theta} &= q \\
\dot{q} &= \frac{M(\alpha, q, \delta)}{I} = M_{\alpha}\alpha + M_{q}q + M_{\delta}\delta \\
\end{align}
$$
where $\delta$ is the canard deflection. The relation between the lift acceleration $a_{L}$ and $\delta$  is
$$
a_{L} = L_{\alpha}\alpha + L_{\delta}\delta;\quad \delta = \frac{a_{L} - L_{\alpha}\alpha}{L_{\delta}}
$$
The relation between the lateral acceleration $a_{M}$ and lift acceleration $a_{L}$ is
$$
a_{M} = a_{L}\cos\alpha
$$
Therefore,
$$
\begin{align}
\dot{\alpha} &= q - \frac{1}{V_{M}}a_{L}\cos{\alpha} \\
\dot{q} &= M_{\alpha}\alpha + M_{q}q + M_{\delta}\frac{a_{L} - L_{\alpha}\alpha}{L_{\delta}} \\
&= \left( M_{\alpha} - M_{\delta}\frac{L_{\alpha}}{L_{\delta}} \right)\alpha + M_{q}q + \frac{M_{\delta}}{L_{\delta}}a_{L}
\end{align}
$$

