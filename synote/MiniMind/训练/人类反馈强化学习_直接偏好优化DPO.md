# 人类反馈强化学习:直接偏好优化DPO

[https://zhuanlan.zhihu.com/p/16460559859](https://zhuanlan.zhihu.com/p/16460559859?utm_medium=social&utm_psn=1895828713637126806&utm_source=wechat_session&scene=1)

[https://zhuanlan.zhihu.com/p/15299083505](https://zhuanlan.zhihu.com/p/15299083505)

# 数学原理

## **二元对比偏好模型（Bradley–Terry）**

当我们有两个候选回答 $y_1, y_2$，假设它们对应的（隐含）“分数”分别为 $r(x,y_1)$ 和 $r(x,y_2)$，Bradley–Terry 模型定义人类偏好 $y_1 \succ y_2$ 的概率为

$$
p^\ast(y_1 \succ y_2 \mid x) \;=\; \frac{\exp\bigl(r(x,y_1)\bigr)}{\exp\bigl(r(x,y_1)\bigr) + \exp\bigl(r(x,y_2)\bigr)}
\;=\; \sigma\bigl(r(x,y_1)-r(x,y_2)\bigr),
$$

其中 $\sigma(z)=1/(1+e^{-z})$ 为 Sigmoid 函数(logistic函数)。

## **二元交叉熵损失（Binary Cross-Entropy）**

若真实偏好标签为 $y=1$ 表示“人类更偏好 $y_1$”，则预测概率为 $p=\sigma(r(x,y_1)-r(x,y_2))$，对应的损失为

$$
\mathcal{L}_{\rm BCE} = -\bigl[y\log p + (1-y)\log(1-p)\bigr]
= -\log\sigma\bigl(r(x,y_1)-r(x,y_2)\bigr)\quad(y=1).
$$

---

## **策略梯度与期望奖励**

在策略 $\pi_\theta(y\mid x)$ 下，强化学习希望最大化期望奖励

$$
J(\theta)
= \mathbb{E}_{x\sim \mathcal{D},\,y\sim \pi_\theta}\bigl[r(x,y)\bigr].
$$

> $\mathbb{E}_{x\sim \mathcal{D},\,y\sim \pi_\theta}\bigl[r(x,y)\bigr]$表示对随机变量 $x$ 和 $y$ 在各自分布下的期望值（expectation）：
>
> * 下标 $x\sim\mathcal{D}$ 说明 $x$ 是从数据分布 $\mathcal{D}$（通常是训练集或环境采样分布）中采样得到的；
> * 下标 $y\sim\pi_\theta$ 说明 在给定 $x$ 的条件下，行动／生成结果 $y$ 是根据策略（policy）$\pi_\theta(y\mid x)$ 随机采样得到的。

利用策略梯度（Policy Gradient）可得

$$
\nabla_\theta J(\theta)
= \mathbb{E}\bigl[r(x,y)\,\nabla_\theta\log\pi_\theta(y\mid x)\bigr].
$$

> **目标函数**
>
> $$
> J(\theta)
> = \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\bigl[r(x,y)\bigr]
> = \sum_x p_\mathcal{D}(x)\sum_y \pi_\theta(y\mid x)\,r(x,y).
> $$
>
> **对参数** **$\theta$** **求梯度**
>
> $$
> \nabla_\theta J(\theta)
> = \nabla_\theta \sum_x p_\mathcal{D}(x)\sum_y \pi_\theta(y\mid x)\,r(x,y).
> $$
>
> 注意到 $p_\mathcal{D}(x)$ 与 $\theta$ 无关，可以提出求和／积分之外：
>
> $$
> = \sum_x p_\mathcal{D}(x)\sum_y \nabla_\theta\bigl[\pi_\theta(y\mid x)\bigr]\;r(x,y).
> $$
>
> **score function 技巧**  
> 对概率密度（或质量） $\pi_\theta(y\mid x)$ 直接求梯度，不易计算，但我们可以写成：
>
> $$
> \nabla_\theta\pi_\theta(y\mid x)
> = \pi_\theta(y\mid x)\,\nabla_\theta\log\pi_\theta(y\mid x).
> $$
>
> 这一步利用了链式法则：
>
> $$
> \nabla_\theta\log\pi = \frac{1}{\pi}\nabla_\theta\pi
> \quad\Longrightarrow\quad
> \nabla_\theta\pi = \pi\,\nabla_\theta\log\pi.
> $$
>
> **带回原式**
>
> $$
> \nabla_\theta J
> = \sum_x p_\mathcal{D}(x)\sum_y
>   \Bigl[\pi_\theta(y\mid x)\,\nabla_\theta\log\pi_\theta(y\mid x)\Bigr]
>   \,r(x,y).
> $$
>
> **恢复期望形式**  
> 上式正是对 $x\sim\mathcal{D}$、\;$y\sim\pi_\theta(\cdot\mid x)$ 的双重期望：
>
> $$
> \boxed{
>   \nabla_\theta J(\theta)
>   = \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}
>     \bigl[r(x,y)\,\nabla_\theta\log\pi_\theta(y\mid x)\bigr].
> }
> $$

## **KL 约束的强化学习目标**

为防止策略偏离预训练模型（参考策略 $\pi_{\rm ref}$），常在目标中加入 KL 散度惩罚：

$$
\max_{\pi_\theta}
\;\mathbb{E}_{y\sim \pi_\theta}\bigl[r(x,y)\bigr]
\;-\;\beta\,D_{\rm KL}\bigl(\pi_\theta(\cdot\mid x)\;\|\;\pi_{\rm ref}(\cdot\mid x)\bigr),
$$

其中

$$
D_{\rm KL}(P\|Q)
= \mathbb{E}_P\bigl[\log P - \log Q\bigr]= \sum_x P(x)\,\bigl(\log P(x)-\log Q(x)\bigr)=\sum_{x}P(x)\,\log\frac{P(x)}{Q(x)}.
$$

对于离散分布，KL 散度定义为

$$
D_{\mathrm{KL}}\bigl(P\|Q\bigr)
=\sum_{x}P(x)\,\log\frac{P(x)}{Q(x)}.
$$

对于连续分布，则写作

$$
D_{\mathrm{KL}}\bigl(P\|Q\bigr)
=\int p(x)\,\log\frac{p(x)}{q(x)}\,\mathrm{d}x.
$$

它度量了「用 $Q$ 来近似 $P$ 时，平均会损失多少信息」。

---

> ### 定义
>
> 对于离散分布，KL 散度定义为
>
> $$
> D_{\mathrm{KL}}\bigl(P\|Q\bigr)
> =\sum_{x}P(x)\,\log\frac{P(x)}{Q(x)}.
> $$
>
> 对于连续分布，则写作
>
> $$
> D_{\mathrm{KL}}\bigl(P\|Q\bigr)
> =\int p(x)\,\log\frac{p(x)}{q(x)}\,\mathrm{d}x.
> $$
>
> 它度量了「用 $Q$ 来近似 $P$ 时，平均会损失多少信息」。
>
> ### 为什么 $D_{\mathrm{KL}}\ge0$
>
> 这一性质通常称为 Gibbs 不等式，可以用 Jensen 不等式来证明。
>
> 1. 设函数 $f(t)=-\log t$。由于 $-\log$ 在 $(0,\infty)$ 上是**凸函数**，对任意正权重分布 $\{w_i\}$ 和正数 $\{t_i\}$，都有
>
>     $$
>     f\Bigl(\sum_i w_i\,t_i\Bigr)
>     \;\le\;
>     \sum_i w_i\,f(t_i).
>     $$
> 2. 令索引 $i$ 对应样本点 $x$，权重 $w_i=P(x)$，令
>
>     $$
>     t_i = \frac{Q(x)}{P(x)}.
>     $$
>
>     代入 Jensen：
>
>     $$
>     -\log\!\Bigl(\sum_x P(x)\,\tfrac{Q(x)}{P(x)}\Bigr)
>     \;\le\;
>     \sum_x P(x)\,\bigl[-\log\tfrac{Q(x)}{P(x)}\bigr].
>     $$
> 3. 注意到 $\sum_x P(x)\,\tfrac{Q(x)}{P(x)}=\sum_x Q(x)=1$，因此左边是 $-\log1=0$。  
>     右边则恰好是
>
>     $$
>     \sum_x P(x)\,\log\frac{P(x)}{Q(x)}
>     =D_{\mathrm{KL}}(P\|Q).
>     $$
> 4. 综合得
>
>     $$
>     0 \;\le\; D_{\mathrm{KL}}(P\|Q).
>     $$
>
> ---
>
> ### 何时等号成立？
>
> 当且仅当在 Jensen 不等式中所有的 $t_i=Q(x)/P(x)$ 相等，也就是对所有 $x$ 满足
>
> $$
> \frac{Q(x)}{P(x)} = \text{常数}
> \quad\Longrightarrow\quad
> P(x)=Q(x)
> $$
>
> （在所有 $P(x)>0$ 的点上）。因此，只有当两个分布**完全相同**时，KL 散度才会等于零。

## **最优策略的软最大化形式**

对上述受 KL 约束的目标，最优解可证明为

$$
\pi_r(y\mid x)
= \frac{1}{Z(x)}\,\pi_{\rm ref}(y\mid x)\,
  \exp\!\Bigl(\tfrac{1}{\beta}r(x,y)\Bigr),\quad
Z(x)=\sum_{y'}\pi_{\rm ref}(y'\mid x)\exp\!\Bigl(\tfrac{1}{\beta}r(x,y')\Bigr).
$$

> ### 一、写出约束优化问题
>
> 对于每个固定的提示 $x$，我们在所有合法的概率分布 $\pi(\cdot\mid x)$ 上，求解
>
> $$
> \begin{aligned}
> &\max_{\pi(\cdot\mid x)} && 
> \mathbb{E}_{y\sim\pi(\cdot\mid x)}\bigl[r(x,y)\bigr]= \sum_{y} \pi(y\mid x)\,r(x,y).
> \\
> &\text{subject to} &&
> \;D_{\rm KL}\bigl(\pi(\cdot\mid x)\|\pi_{\rm ref}(\cdot\mid x)\bigr)=\sum_{y}\pi(y\mid x)\,\log\frac{\pi(y\mid x)}{\pi_{\rm ref}(y\mid x)}
> \;\le\;\varepsilon,
> \\
> & &&\sum_{y}\pi(y\mid x)=1,\quad
> \pi(y\mid x)\ge0\;\;\forall y.
> \end{aligned}
> $$
>
> ---
>
> ### 二、构造拉格朗日函数
>
> 引入拉格朗日乘子 $\lambda\ge0$ 对应 KL 约束，乘子 $\eta$ 对应“归一化”约束，写出拉格朗日函数
>
> $$
> \begin{aligned}
> \mathcal{L}(\pi,\lambda,\eta)\;=\;&
> \sum_{y}\pi(y)r(x,y)
> \;-\;\lambda\Bigl(\sum_{y}\pi(y)\log\tfrac{\pi(y)}{\pi_{\rm ref}(y)}-\varepsilon\Bigr)
> \\
> &\;+\;\eta\Bigl(\sum_{y}\pi(y)-1\Bigr),
> \end{aligned}
> $$
>
> 为了简洁，在公式里省略了“$\mid x$”标记。
>
> ---
>
> ### 三、对 $\pi(y)$ 求驻点（KKT 一阶条件）
>
> 对每个 $y$ 求偏导并令其为 0：
>
> $$
> \frac{\partial\mathcal{L}}{\partial\pi(y)}
> = r(x,y)
> \;-\;\lambda\Bigl(\log\pi(y)-\log\pi_{\rm ref}(y)+1\Bigr)
> \;+\;\eta
> \;=\;0.
> $$
>
> 整理得
>
> $$
> \log\pi(y)
> =\;\frac{1}{\lambda}\,r(x,y)
> \;+\;\log\pi_{\rm ref}(y)
> \;-\;1
> \;-\;\frac{\eta}{\lambda}.
> $$
>
> ---
>
> ### 四、解出最优分布形式
>
> 上式两边取指数，并将与 $y$ 无关的常数收入归一化常数 $Z(x)$，可得
>
> $$
> \pi(y)
> \;\propto\;
> \pi_{\rm ref}(y)\,
> \exp\!\Bigl(\tfrac{1}{\lambda}\,r(x,y)\Bigr).
> $$
>
> 令 $\beta=\lambda$，并写出显式的归一化：
>
> $$
> \boxed{
> \pi_r(y\mid x)
> =\frac{1}{Z(x)}\,\pi_{\rm ref}(y\mid x)\,
>   \exp\!\Bigl(\tfrac{1}{\beta}r(x,y)\Bigr),
> \quad
> Z(x)=\sum_{y'}\pi_{\rm ref}(y'\mid x)\,\exp\!\Bigl(\tfrac{1}{\beta}r(x,y')\Bigr).
> }
> $$
>
> ---
>
> ### 五、等价的无约束形式
>
> 实际上，上述带约束的优化等价于无约束地最大化拉格朗日函数（或等价地，最大化）
>
> $$
> \mathbb{E}_{y\sim\pi_\theta}\bigl[r(x,y)\bigr]
> \;-\;\beta\,
> D_{\rm KL}\bigl(\pi_\theta(\cdot\mid x)\|\pi_{\rm ref}(\cdot\mid x)\bigr).
> $$
>
> 对这个目标直接求驻点，同样会得到完全相同的解析解。

## **隐式奖励重参数化**

DPO 的关键在于**直接**以策略 $\pi_\theta$ 的对数概率比重参数化奖励，从而把原本的 RLHF 问题转化为一个二元交叉熵问题：

定义

$$
\hat r_\theta(x,y)
\;=\;
\beta\,
\log\frac{\pi_\theta(y\mid x)}{\pi_{\rm ref}(y\mid x)}.
$$

这意味着策略本身就隐式地定义了“奖励”函数。

## **DPO 损失**

给定人类偏好数据集 $\{(x^{(i)},y_w^{(i)},y_\ell^{(i)})\}$，DPO 目标为最小化

$$
\mathcal{L}_{\rm DPO}(\theta)
= -\mathbb{E}_{(x,y_w,y_\ell)}\!
\Bigl[\log\sigma\bigl(
  \hat r_\theta(x,y_w)
  - \hat r_\theta(x,y_\ell)
\bigr)\Bigr].
$$

展开即

$$
\boxed{
\mathcal{L}_{\rm DPO}
= -\mathbb{E}\Bigl[\log\sigma\bigl(
    \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)}
    - \beta\log\frac{\pi_\theta(y_\ell|x)}{\pi_{\rm ref}(y_\ell|x)}
  \bigr)\Bigr].
}
$$

## **梯度解读**

$$
\nabla_\theta\mathcal{L}_{\rm DPO}
= -\beta\,\mathbb{E}\Bigl[
  \sigma\bigl(\Delta\hat r\bigr)\,\bigl(
    \nabla_\theta\log\pi_\theta(y_w|x)
    - \nabla_\theta\log\pi_\theta(y_\ell|x)
  \bigr)\Bigr],
$$

其中 $\Delta\hat r=\hat r_\theta(x,y_\ell)-\hat r_\theta(x,y_w)$。可见，DPO 在增大“更优”回答 $y_w$ 概率的同时，减小“不优”回答 $y_\ell$ 的概率，且权重由 $\sigma(\Delta\hat r)$ 动态调整，避免模型退化。

‍

# 代码分析

## `dpo_loss`​

$$
\mathcal{L}_{\rm DPO}
=-\mathbb{E}_{(x,y_w,y_\ell)}\Bigl[\log\sigma\bigl(
\beta\,( \underbrace{\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)}}_{A}
- \underbrace{\log\tfrac{\pi_\theta(y_\ell|x)}{\pi_{\rm ref}(y_\ell|x)}}_{B}
)\bigr)\Bigr]
$$

```python
def dpo_loss(ref_probs, probs, mask, beta):
    # 1. mask.sum 得到每个样本的真实长度，用于后续归一化“序列级别的 log-prob”
    seq_lengths = mask.sum(dim=1, keepdim=True)

    # 2. 按 token 累加 log-probs，再除以长度，得到每个序列的平均 log-prob
    #    ref_probs 与 probs 应该是形状 [batch, seq_len] 的 log π_ref 和 log π_θ
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs     = (    probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 3.Batch 里的前一半是“更优”回答 y_w，后一半是“次优”回答 y_ℓ
    batch_size           = ref_probs.shape[0]
    chosen_ref_probs     = ref_probs[:batch_size // 2]   # log π_ref(y_w|x)
    reject_ref_probs     = ref_probs[batch_size // 2:]   # log π_ref(y_ℓ|x)
    chosen_probs         = probs    [:batch_size // 2]   # log π_θ(y_w|x)
    reject_probs         = probs    [batch_size // 2:]   # log π_θ(y_ℓ|x)

    # 4. 计算两个 log-ratio：
    #    A = log π_θ(y_w) - log π_θ(y_ℓ)
    #    B = log π_ref(y_w) - log π_ref(y_ℓ)
    pi_logratios  = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs

    # 5. DPO 核心：用 A – B 作为 sigmoid 的输入
    #    logits = A - B = [log π_θ(y_w) - log π_ref(y_w)] 
    #                  - [log π_θ(y_ℓ) - log π_ref(y_ℓ)]
    logits = pi_logratios - ref_logratios

    # 6. 负对数 sigmoid，就是 -log σ(β·logits)
    #    完全对应公式里的 -log σ(β (A–B))
    loss = -F.logsigmoid(beta * logits)

    # 7. 对全batch取平均
    return loss.mean()
```

* **第2步** 把 `ref_probs`​、`probs`​ 从 token 维度上求和并除以长度，相当于把序列的 log‑likelihood 归一化为“平均 log‑prob”，也就是

  $$
  \log\pi_\theta(y\mid x)\;=\;\tfrac1{L}\sum_{t=1}^L\log\pi_\theta(y_t\mid x,y_{<t})
  $$
* **第4步** 中的 `pi_logratios`​ 就是公式里 $A=\log\pi_\theta(y_w)-\log\pi_\theta(y_\ell)$，  
  ​`ref_logratios`​ 是 $B=\log\pi_{\rm ref}(y_w)-\log\pi_{\rm ref}(y_\ell)$。
* **第5步** `logits = A – B`​ 恰好等于

  $$
  \underbrace{\log\frac{\pi_\theta(y_w)}{\pi_{\rm ref}(y_w)}}_{\hat r_\theta(x,y_w)}
    \;-\;
    \underbrace{\log\frac{\pi_\theta(y_\ell)}{\pi_{\rm ref}(y_\ell)}}_{\hat r_\theta(x,y_\ell)}.
  $$
* **第6步** 计算 $-\log\sigma\bigl(\beta\cdot\text{logits}\bigr)$，即 DPO 的单样本损失：

  $$
  -\log\sigma\!\bigl(\beta(\hat r_\theta(x,y_w)-\hat r_\theta(x,y_\ell))\bigr).
  $$

整体上，这段代码准确地实现了 DPO 的核心公式，通过最小化该 loss，可以让模型 $\pi_\theta$ 更倾向于生成人类偏好的 $y_w$。

‍
