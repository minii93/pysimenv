## Dyn6DOF

##### 2022/08/10

#### Mathematical model

Each variable is defined as

* $m$: mass
* $J$: moment of inertia
* $p^{b}$: position in inertial axes
* $v^{i}$: velocity in inertial axes
* $v^{b}$: velocity in body axes
* $q_{b}^{i}$: unit quaternion representing the rotation from the body frame to the inertial frame
* $R_{b}^{i}$: rotation matrix representing the rotation from the body frame to the inertial frame
* $\omega_{b/i}^{b}$: angular velocity (body rotation rates) in body exes
* $f^{b}$: applied force in body axes
* $m^{b}$: applied moment in body axes

The translational kinematics and dynamics are given as
$$
\begin{align}
\frac{d}{dt}p^{b} &= R_{b}^{i}v^{b} \\
\frac{d}{dt}v^{b} &= -\omega_{b/i}^{b} \times v^{b} + \frac{1}{m} f^{b}
\end{align}
$$
Note that $v^{i} = R_{b}^{i}v^{b}$.

The rotational kinematics and dynamics are given as
$$
\begin{align}
\frac{d}{dt}q_{b}^{i} &= \frac{1}{2}q\otimes \begin{bmatrix} 0 \\ \omega_{b/i}^{b} \end{bmatrix} \\
&= \frac{1}{2} \begin{bmatrix}
q_{0} & -q_{1} & -q_{2} & -q_{3} \\
q_{1} & q_{0} & -q_{3} & q_{2} \\
q_{2} & q_{3} & q_{0} & -q_{1} \\
q_{3} & -q_{2} & q_{1} & q_{0}
\end{bmatrix} \begin{bmatrix} 0 \\ \omega_{b/i}^{b} \end{bmatrix} \\
\frac{d}{dt}\omega_{b/i}^{b} &= J^{-1} \left( - \omega_{b/i}^{b} \times (J \omega_{b/i}^{b}) + m^{b} \right)
\end{align}
$$
Denote the scalar component and vector component of $q_{b}^{i}$ as $\eta$, $\epsilon$, respectively. That is, $q_{b}^{i}=[\eta \: \epsilon^{T}]^{T}$. The relation between $q_{b}^{i}$ and $R_{b}^{i}$ can be expressed as
$$
R_{b}^{i} = I + 2\eta S(\epsilon) + 2S(\epsilon)^{2}
$$
where $S(x)$ is the matrix representation of the vector cross product.
$$
S(x) = \begin{bmatrix}
0 & -x_{3} & x_{2} \\
x_{3} & 0 & -x_{1} \\
-x_{2} & x_{1} & 0
\end{bmatrix},\quad x \in \mathbb{R}^{3}
$$
