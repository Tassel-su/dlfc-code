# 第 15 章：离散潜变量（K-means / GMM / EM）

> 第 15 章（印刷页 459-494）。脚本：`ch15_discrete_latent.py`

## 核心内容
- **K-means**：硬指派（E 步）+ 中心更新（M 步）交替；k-means++ 初始化
- **GMM**：p(x) = Σπ_k N(x|μ_k,Σ_k)，MLE 无闭式解 -> EM
- **EM**：E 步算责任 γ_nk（后验），M 步更新 π/μ/Σ
- **ELBO**：证据下界单调上升 = EM 收敛保证
- 图像分割：像素强度聚类

## 运行结果（可直接复现）
- K-means 中心匹配率 **1.00**（k-means++ 初始化）
- EM：ELBO 从 -1554 单调上升到 **-758.7**，π=[0.333,0.333,0.333]（真实等权重）
- MNIST 图像分割为 4 个灰度区

## 关键公式
- K-means 目标：min Σ_n ||x_n - μ_{z_n}||²
- GMM 似然：p(X|θ) = Π_n Σ_k π_k N(x_n|μ_k,Σ_k)
- EM E 步：γ_nk = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)
- EM M 步：μ_k = Σγ_nk x_n / N_k，Σ_k = Σγ_nk (x_n-μ_k)(x_n-μ_k)ᵀ/N_k
- ELBO：Σ_n Σ_k γ_nk [ln π_k + ln N(x_n|μ_k,Σ_k) - ln γ_nk]

## 自测题
1. K-means 与 EM 的关系？
2. ELBO 单调性为什么是 EM 的收敛保证？
3. GMM 为什么不能用普通梯度上升直接优化？

<details><summary>答案</summary>
1. K-means 是 GMM 取各向同性协方差且 σ→0 的极限：责任退化为硬指派。
2. EM 每次迭代都使 ELBO 不减（E 步是下界的紧化，M 步最大化下界），
   而 ELBO ≤ log p(X)，因此迭代单调上升并收敛。
3. 对数似然含 Σ_k 的对数，耦合严重；EM 引入隐变量后每步都有闭式解，
   更稳定高效。
</details>
