## Local linear Factorization


$$
\hat{y} =\sum_{v\in C} \gamma_v(x)\{W_0(v) + W(v)x + \frac{1}{2}\sum_{f=1}^{k}[(\sum_{j=1}^{p}V(v)_{j,f}x_j)^2-\sum_{j=1}^{p}V(v)_{j,f}^2x_j^2]\}
$$

其中：

* C : set of k-nearest anchor points

* $$\gamma_{v_j}(x)$$ : coefficients of sample $x$ to anchor point $v_j$

  * Euclidean distance  

  $$
  \gamma_{v_j}(x) =\frac{d(x,v_j)}{\sum_{v_j \in C}d(x,v_j)}
  $$





* W, V：
  * parameters of FM

## objectives

$$
l = \frac{1}{2}(\hat{y}(x|\theta)-y)^2 + \lambda\sum\theta^2
$$

## SGD

* partial derivative

  * $$
    \frac{\partial\hat{y}(x)}{\partial\theta}=\begin{cases} \gamma(x)_j  & if \ \theta \ is \ W_{j0} \\ \gamma(x)_jx_l &  if \ \theta \ is \ W_{jl} \\ \gamma(x)_jx_l \sum_{j \neq l}v_{j,f}x_j & if \ \theta \ is V_{j,l,f}\end{cases}
    $$





* Update

$$
\theta  = \theta - \eta(\frac{\partial\hat{y}}{\partial\theta} + 2\lambda\theta)
$$

