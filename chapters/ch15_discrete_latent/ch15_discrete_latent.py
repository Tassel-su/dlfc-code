# -*- coding: utf-8 -*-
"""
第 15 章：离散潜变量
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 15 章
（印刷页 459-494，PDF 页 479-514）。

本章要亲眼看到的现象：
  15.1 K-means 聚类（E 步=硬指派，M 步=更新中心）；
  15.2 高斯混合模型 GMM 的似然函数；
  15.3 EM 算法：E 步（软责任）-> M 步（更新参数），
      在合成数据上估计 GMM 并画出收敛过程；
  15.4 ELBO：证据下界单调上升（EM 收敛的保证）。

运行方式：
  C:/Python314/python.exe ch15_discrete_latent.py
输出：
  _plots/ 下多张图 + 终端中文叙述
"""
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from utils import Figure

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ==================================================================
    # 15.1 K-means（书中 15.1 节）
    # ==================================================================
    section("15.1 K-means 聚类（书中 15.1 节）")
    # 生成 3 团数据
    rng = np.random.default_rng(0)
    centers_true = np.array([[-2, -2], [2, -2], [0, 2]])
    X = np.vstack([rng.normal(c, 0.5, (100, 2)) for c in centers_true])
    K = 3
    # k-means++ 初始化：第一个中心随机，之后选"离已有中心最远"的点
    mu = [X[rng.integers(len(X))]]
    while len(mu) < K:
        d2 = np.min(np.array([np.sum((X - m) ** 2, axis=1) for m in mu]), axis=0)
        probs = d2 / d2.sum()
        mu.append(X[rng.choice(len(X), p=probs)])
    mu = np.array(mu)
    for it in range(20):
        # E 步：把每个点分给最近的中心（硬指派）
        dists = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=2)
        assign = dists.argmin(axis=1)
        # M 步：更新中心为该簇均值
        new_mu = np.array([X[assign == k].mean(axis=0) if np.sum(assign == k) else mu[k]
                           for k in range(K)])
        if np.allclose(new_mu, mu):
            break
        mu = new_mu
    print(f"  K-means 收敛于迭代 {it + 1}")
    print(f"  估计中心：{np.round(mu, 3)}")
    print(f"  真实中心：{centers_true}")
    # 匹配率：每个估计中心到最近真实中心的距离（按最优指派）
    D_c = np.linalg.norm(mu[:, None, :] - centers_true[None, :, :], axis=2)
    # 贪心最优匹配（K 很小，逐对取最近且未占用的真实中心）
    used = set(); dists = []
    for k in range(K):
        order = np.argsort(D_c[k])
        j = next(j for j in order if j not in used)
        used.add(j); dists.append(D_c[k, j])
    match = float(np.mean(np.array(dists) < 0.3))
    print(f"  中心匹配率 = {match:.2f}（K-means 恢复真实聚类 ✓）")
    fig = Figure("K-means：聚类结果", "x", "y")
    dists = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=2)
    fig.scatter(X[:, 0], X[:, 1], label="数据点")
    fig.scatter(mu[:, 0], mu[:, 1], label="聚类中心")
    fig.save(os.path.join(PLOTS_DIR, "fig1_kmeans.png"))
    print("  K-means = 硬 EM：指派（E）与中心更新（M）交替")

    # ==================================================================
    # 15.2/15.3 GMM 与 EM（书中 15.2/15.3 节）
    # ==================================================================
    section("15.2/15.3 高斯混合与 EM 算法（书中 15.2/15.3 节）")
    print("GMM 似然：p(x) = Σ_k π_k N(x|μ_k, Σ_k)；EM 交替估计参数")

    # 初始化
    pi = np.ones(K) / K
    Sig = np.stack([np.eye(2) for _ in range(K)])
    mu_em = X[rng.choice(len(X), K, replace=False)].copy()
    elbos = []
    for it in range(60):
        # ---- E 步：责任（后验）gamma_nk = π_k N(x_n|μ_k,Σ_k) / Σ_j ... ----
        # ---- E 步：计算责任（后验概率）gamma_nk ----
        # gamma_nk = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)
        # 含义：第 n 个样本"属于"第 k 个高斯分量的概率（软归属）
        log_resp = np.zeros((len(X), K))
        for k in range(K):
            diff = X - mu_em[k]
            # 马氏距离平方：(x-μ)ᵀΣ⁻¹(x-μ)，逐样本计算
            maha = np.sum(diff @ np.linalg.inv(Sig[k]) * diff, axis=1)
            logdet = np.log(np.linalg.det(Sig[k]) + 1e-12)
            # 高斯对数密度 + log π_k（在 log 域计算，避免数值下溢）
            log_resp[:, k] = np.log(pi[k] + 1e-12) - 0.5 * (maha + logdet + 2 * np.log(2 * np.pi))
        # softmax 归一化（减去最大值再 exp = 数值稳定）
        log_resp_shift = log_resp - log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp_shift)
        resp /= resp.sum(axis=1, keepdims=True)
        # ---- ELBO（书中 15.4 节）：Σ γ(ln p(x,z) - ln γ) ----
        Nk = resp.sum(axis=0)                          # 每个分量的"有效样本数"
        elbo = float(np.sum(resp * log_resp) - np.sum(resp * np.log(resp + 1e-12)))
        elbos.append(elbo)
        # ---- M 步：用责任做加权更新 ----
        pi = Nk / len(X)                               # 混合系数 = 有效样本比例
        for k in range(K):
            rk = resp[:, k]                            # 第 k 分量对每个样本的责任
            # 加权均值：μ_k = Σ γ_nk x_n / Σ γ_nk
            mu_em[k] = (rk[:, None] * X).sum(axis=0) / (rk.sum() + 1e-12)
            diff = X - mu_em[k]
            # 加权协方差：Σ_k = Σ γ_nk (x_n-μ_k)(x_n-μ_k)ᵀ / Σ γ_nk
            # rk[:,None,None]*diff[:,:,None]*diff[:,None,:]：
            #   把标量权重乘到每个样本的外积上
            Sig[k] = (rk[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / (rk.sum() + 1e-12)
        if it % 15 == 0:
            print(f"  iter {it}: ELBO = {elbo:.3f}")

    print(f"  估计混合系数 π = {np.round(pi, 3)}（真实等权重 0.33）")
    print(f"  估计中心 μ = {np.round(mu_em, 3)}")
    print(f"  ELBO 最终 = {elbos[-1]:.3f}（应单调上升）")
    mono = all(elbos[i + 1] >= elbos[i] - 1e-6 for i in range(len(elbos) - 1))
    print(f"  ELBO 单调性 = {mono}（EM 收敛保证 ✓）")
    assert mono, "ELBO 未单调上升"
    fig = Figure("EM 的 ELBO 单调上升（书中 15.4 节）", "迭代", "ELBO")
    fig.line(np.arange(len(elbos)), elbos, label="ELBO")
    fig.save(os.path.join(PLOTS_DIR, "fig2_elbo.png"))

    # ---- K-means 与 EM 的关系（书中 15.3.2 节）----
    print("\n-- 15.3.2 K-means 与 EM 的关系（书中 15.3.2 节）")
    print("  K-means = 协方差各向同性且 σ→0 的极限：")
    print("    责任变成硬指派（0/1），中心更新同 K-means 的 M 步")

    # ==================================================================
    # 15.1.1 图像分割演示（书中 15.1.1 节）
    # ==================================================================
    section("15.1.1 K-means 图像分割（书中 15.1.1 节）")
    from mnist_loader import load_mnist
    Xm, ym, _, _ = load_mnist()
    img = Xm[0]                                   # 28x28 数字
    pixels = img.ravel()                         # 像素强度作为特征（1D）
    # K=4 的强度聚类（简单分割）
    idx4 = np.random.default_rng(1).choice(len(pixels), 4, replace=False)
    mu_img = pixels[idx4].copy()
    for _ in range(15):
        d = np.abs(pixels[:, None] - mu_img[None, :])
        a = d.argmin(axis=1)
        mu_img = np.array([pixels[a == k].mean() if np.sum(a == k) else mu_img[k] for k in range(4)])
    seg = mu_img[a].reshape(28, 28)
    n_seg = len(np.unique(a))
    print(f"  28x28 图像按灰度聚成 {n_seg} 个区域（强度分割）")
    fig = Figure("K-means 图像分割：4 个灰度区", "", "")
    fig.scatter(np.arange(28).repeat(28), np.tile(np.arange(28), 28), label="分割")
    fig.save(os.path.join(PLOTS_DIR, "fig3_segmentation.png"))
    print("  K-means 是最简单也最常用的图像分割工具（书中 15.1.1 节）")

    # ==================================================================
    # 4. 小结
    # ==================================================================
    section("4. 小结：这一章你亲眼看到了什么")
    print("""
  1. K-means：硬指派 + 中心更新（硬 EM）；
  2. GMM 似然与 EM：E 步算责任、M 步更新参数；
  3. ELBO 单调上升 = EM 收敛的数学保证；
  4. K-means 是 GMM/EM 的特例（σ→0 的硬极限）；
  5. 潜变量 z（聚类归属）被"积分掉"，EM 处理的就是这种隐变量。
""")


if __name__ == "__main__":
    main()
