## Three-loop autopilot

##### 2022/10/17

#### Pitch dynamic model

Assume that a missile configuration is symmetric and a roll motion is stabilized. The longitudinal dynamics is given by
$$
\begin{align}
\dot{\alpha} &= q + \frac{a_{z}}{V} \\
\dot{q} &= \frac{QSd}{I_{yy}}\left( C_{m_{0}} + C_{m_{q}}\left(\frac{d}{2V} \right)q + C_{m_{\delta}}\delta \right) \\
\end{align}
$$
where
$$
a_{z} = \frac{QS}{m}\left(C_{z_{0}} + C_{z_{\delta}}\delta \right)
$$
The linearized aerodynamic model is given by
$$
\begin{align}
\dot{\alpha} &= q + \frac{a_{z}}{V} = q - \frac{a_{L}}{V} \\
\dot{q} &= \frac{QSd}{I_{yy}}\left(C_{m_{0, \alpha}}\alpha + C_{m_{q}}\left( \frac{d}{2V} \right)q + C_{m_{\delta}}\delta \right) = M_{\alpha}\alpha + M_{q}q + M_{\delta}\delta \\
a_{L} &= -\frac{QS}{m}(C_{z_{0, \alpha}}\alpha + C_{z_{\delta}}\delta) = L_{\alpha}\alpha + L_{\delta}\delta
\end{align}
$$


#### Three-loop autopilot

The dynamic of the lift acceleration can be approximated as
$$
\begin{align}
\dot{a_{L}} &= L_{\alpha}\dot{\alpha} + L_{\delta}\dot{\delta} \approx L_{\alpha}\dot{\alpha} \\
&= L_{\alpha}\left(q - \frac{a_{L}}{V} \right)
\end{align}
$$
since generally $\vert  L_{\delta} \vert \ll \vert L_{\alpha} \vert$ holds. For the acceleration loop (outer loop), the desired pitch angle is computed as
$$
\begin{align}
q_{c} &= \frac{1}{L_{\alpha}}\frac{1}{\tau_{d}}(a_{L_{c}} - a_{L}) + \frac{1}{V}a_{L} \\
&= \left( \frac{1}{L_{\alpha}}\frac{1}{\tau_{d}} - \frac{1}{V} \right) \left( \frac{V}{V - L_{\alpha}\tau_{d}}a_{L_{c}} - a_{L} \right) \\
&= K_{A}(K_{DC}a_{L_{c}} - a_{L})
\end{align}
$$
For the pitch rate loop (inner loop), the desired fin command is computed as
$$
\begin{align}
\delta_{c} &= \frac{1}{M_{\delta}}\left( \omega_{d}^{2}\int (q_{c} - q)dt - 2\zeta_{d}\omega_{d}q - M_{q}q \right) \\
&= \frac{1}{M_{q}} \left( \omega_{d}^{2}\int (q_{c} - q)dt - (2\zeta_{d}\omega_{d} + M_{q})q \right) \\
&= \frac{2\zeta_{d}\omega_{d} + M_{q}}{M_{\delta}}\left( \frac{\omega_{d}^{2}}{2\zeta_{d}\omega_{d} + M_{q}}\int (q_{c} - q)dt - q \right) \\
&= K_{R}\left(\omega_{i}\int (q_{c} - q)dt - q \right)
\end{align}
$$
