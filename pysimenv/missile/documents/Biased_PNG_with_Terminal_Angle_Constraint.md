## Biased PNG with Terminal-Angle Constraint

##### 2022/07/26

Contents of this document is based on the following reference:

B. Park, T. Kim, and M. Tahk, "Biased PNG With Terminal-Angle Constraint for Intercepting Nonmaneuvering Targets Under Physical Constrasints"



#### Reference value

Let $\theta_{M}, \theta_{T}$ denote flight path angles of the missile and target, respectively, and $\lambda$ denote the line-of-sight (LOS) angle. The desired terminal flight path angle of the missile is denoted as $\theta_{M_{f}}$. Let us define shifted angles as
$$
\begin{align}
\bar{\theta}_{M_{0}} &= \theta_{M_{0}} - \lambda_{0} \\
\bar{\theta}_{M_{f}} &= \theta_{M_{f}} - \lambda_{0} \\
\bar{\theta}_{T} &= \theta_{T} - \lambda_{0}
\end{align}
$$
The reference integral value of the bias is formulated as
$$
B_{ref}=\bar{\theta}_{M_{f}} - \bar{\theta}_{M_{0}} - N\tan^{-1}\left( \frac{V_{M}\sin\bar{\theta}_{M_{f}} - V_{T}\sin\bar{\theta}_{T}}{V_{M}\cos\bar{\theta}_{M_{f}} - V_{T}\cos\bar{\theta}_{T}} \right)
$$


#### Guidance law

The guidance law is designed as
$$
a_{M}=V_{M}(N\dot{\lambda} + b)
$$
where $a_{M}$ is the acceleration of the missile, $b$ is the bias acceleration. The integral of the bias can be expressed as
$$
B(t) = \int_{0}^{t}b(\tau)\,d\tau
$$
In the impact angle control phase (BPNG; biased proportional navigation guidance), the bias is designed as
$$
b = \frac{1}{\tau}(B_{ref} - B)
$$
where $\tau$ is the time constant.

In the look angle control phase (DPP; deviated pure pursuit), the bias is designed as
$$
b = (1 - N)\dot{\lambda}
$$
BPNG is used until $\sigma$ reaches the maximum look angle $\sigma_{\max}
$. Then, the guidance law is switched to DPP. The guidance law is again switched to BPNG when the condition $\vert b_{BPNG} \vert \le \vert b_{DPP} \vert$ is met.