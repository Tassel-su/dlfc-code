# -*- coding: utf-8 -*-
"""
第 16 章：连续潜变量
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 16 章
（印刷页 495-532，PDF 页 515-552）。

本章要亲眼看到的现象：
  16.1 PCA：
      - 最大方差公式（协方差特征分解）；
      - 最小误差公式（投影-重构）；
      - 数据压缩（降维 + 重构误差）；
      - 白化（whitening）；
      - 高维数据：MNIST 降到 2D 可视化；
  16.2 概率潜变量模型（PPCA）：x = Wz + μ + ε 的生成式视角，
      MLE 与 PCA 的关系；
  16.3 ELBO：连续潜变量下界。

运行方式：
  C:/Python314/python.exe ch16_continuous_latent.py
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
    # 16.1 PCA（书中 16.1 节）
    # ==================================================================
    section("16.1 PCA：最大方差与最小误差（书中 16.1.1/16.1.2 节）")
    # 生成 2D 数据：沿某方向方差大
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, 2000)
    X = np.stack([z, 0.5 * z + rng.normal(0, 0.5, 2000)], axis=1)  # 相关性数据
    X = X - X.mean(axis=0)                            # 中心化

    # 最大方差公式：协方差矩阵特征分解
    S = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(S)              # 升序
    order = np.argsort(eigvals)[::-1]                 # 降序
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    var_ratio = eigvals / eigvals.sum()
    print(f"  协方差特征值：{np.round(eigvals, 3)}（方差占比 {np.round(var_ratio, 3)}）")
    print(f"  第一主成分方向：{np.round(eigvecs[:, 0], 3)}（最大方差方向）")
    print("  主成分 = 数据方差最大的正交方向 —— 最大方差公式")

    # 最小误差公式：投影到 PC1 再重构
    proj1 = X @ eigvecs[:, 0]
    recon = np.outer(proj1, eigvecs[:, 0])
    recon_err = float(np.mean(np.sum((X - recon) ** 2, axis=1)))
    print(f"  用 1 个主成分重构的均方误差 = {recon_err:.3f}（= 被舍弃的特征值 {eigvals[1]:.3f} ✓）")
    assert abs(recon_err - eigvals[1]) < 0.05, "重构误差应等于被舍弃的特征值"  # E||x-x̂||² = Σ被舍弃λ
    fig = Figure("PCA：第一主成分方向（最大方差）", "x1", "x2")
    fig.scatter(X[::5, 0], X[::5, 1], label="数据")
    fig.save(os.path.join(PLOTS_DIR, "fig1_pca.png"))
    print("  PCA 重构误差 = 被舍弃的特征值之和 —— 最小误差公式")

    # ---- 16.1.4 白化 ----
    print("\n-- 16.1.4 白化：把数据变成单位方差、去相关（书中 16.1.4 节）")
    X_white = X @ eigvecs / np.sqrt(eigvals + 1e-9)   # 旋转 + 缩放
    print(f"  白化后协方差 = {np.round(np.cov(X_white.T), 3)}（应≈单位阵）")
    assert np.allclose(np.cov(X_white.T), np.eye(2), atol=0.1), "白化失败"

    # ---- 16.1.5 高维：MNIST 2D 投影 ----
    section("16.1.5 高维数据：MNIST 投影到 2D（书中 16.1.5 节）")
    from mnist_loader import load_mnist
    Xm, ym, _, _ = load_mnist()
    Xsub, ysub = Xm[:2000].reshape(2000, -1), ym[:2000]
    Xsub = Xsub - Xsub.mean(axis=0)
    # 大矩阵用 SVD（比协方差特征分解数值稳定）
    U, s, Vt = np.linalg.svd(Xsub, full_matrices=False)
    proj2 = Xsub @ Vt[:2].T                            # 前两个主成分
    fig = Figure("MNIST 2000 张图 PCA 到 2D：数字自然分簇！", "PC1", "PC2")
    fig.scatter(proj2[:, 0], proj2[:, 1], label="数字")
    fig.save(os.path.join(PLOTS_DIR, "fig2_mnist_pca.png"))
    var2 = (s[:2] ** 2).sum() / (s ** 2).sum()
    print(f"  前 2 个主成分解释方差比例 = {var2:.3f}")
    print("  无监督的 PCA 已经让不同数字分开 —— 潜变量的力量！")

    # ==================================================================
    # 16.2 概率 PCA（书中 16.2 节）
    # ==================================================================
    section("16.2 概率 PCA：x = Wz + μ + ε（书中 16.2.1/16.2.3 节）")
    print("生成式模型：z ~ N(0,I)，x = Wz + μ + ε，ε ~ N(0, σ²I)")
    # 用 EM 估计 W（书中 16.3.2 节 EM for PCA 的核心迭代）
    D = X.shape[1]                                     # 数据维度 2
    M_lat = 1                                          # 潜变量维度 1
    W = rng.normal(0, 0.5, (D, M_lat))
    sigma2 = 1.0
    for _ in range(50):
        # E 步：后验 q(z|x) 的充分统计量
        M_mat = W.T @ W + sigma2 * np.eye(M_lat)
        Minv = np.linalg.inv(M_mat)
        Ez = X @ W @ Minv                              # E[z|x]
        Ezz = Minv + Ez[:, None, :] * Ez[:, :, None]   # E[zzᵀ|x]
        # M 步：更新 W 和 σ²
        W_new = (np.sum([np.outer(X[i], Ez[i]) for i in range(len(X))], axis=0)
                 @ np.linalg.inv(np.sum(Ezz, axis=0)))
        sigma2 = float(np.mean([np.sum(X[i] ** 2) - 2 * X[i] @ W_new @ Ez[i]
                                + np.trace(Ezz[i] @ W_new.T @ W_new)
                                for i in range(len(X))]) / D)
        W = W_new
    # 与 PCA 主方向对比
    cos_sim = abs(float(W[:, 0] @ eigvecs[:, 0]) / (np.linalg.norm(W[:, 0]) * np.linalg.norm(eigvecs[:, 0])))
    print(f"  PPCA 的 W 与 PCA 主成分方向余弦相似度 = {cos_sim:.3f}（方向一致 ✓）")
    assert cos_sim > 0.9, "PPCA 与 PCA 方向不一致"
    print("  概率 PCA 的 MLE 解 = PCA 的主子空间（书中 16.2.3 节结论）")

    # ==================================================================
    # 16.3 ELBO（书中 16.3 节）
    # ==================================================================
    section("16.3 ELBO：连续潜变量下界（书中 16.3 节）")
    print("log p(x) >= ELBO = E_q[log p(x,z)] - E_q[log q(z)]（书中 15.4/16.3 节）")
    # 用简单高斯例子验证 ELBO 分解
    # p(x|z) = N(x; z, 1)，p(z) = N(z; 0, 1)，q(z) = N(z; m, v)（变分近似）
    # 精确后验 p(z|x) = N(m*, v*)，ELBO 在 q=精确后验时取等号
    x_val = 1.5
    v_star = 0.5                                       # 精确后验方差 1/(1+1)=0.5
    m_star = x_val / 2                                 # 精确后验均值 x/(1+1)
    def elbo(m, v):
        # E_q[log p(x,z)] = E_q[log p(x|z)] + E_q[log p(z)]
        e1 = -0.5 * (np.log(2 * np.pi) + (x_val - m) ** 2 + v)      # log p(x|z)
        e2 = -0.5 * (np.log(2 * np.pi) + m ** 2 + v)                # log p(z)
        ent = 0.5 * (np.log(2 * np.pi * v) + 1)                     # 熵 H(q)
        return float(e1 + e2 + ent)
    elbo_opt = elbo(m_star, v_star)
    elbo_bad = elbo(0.0, 1.0)                          # 较差近似
    exact_logp = -0.5 * (np.log(2 * np.pi * 2) + x_val ** 2 / 2)    # log N(x;0,2)
    print(f"  精确 log p(x) = {exact_logp:.4f}")
    print(f"  ELBO（最优 q）= {elbo_opt:.4f}（应等于精确值）")
    print(f"  ELBO（差 q）= {elbo_bad:.4f}（下界更松）")
    assert abs(elbo_opt - exact_logp) < 1e-6, "最优 ELBO 应等于 log p(x)"
    assert elbo_bad < elbo_opt, "差近似应给出更低下界"
    print("  ELBO = 变分推断的核心：最大化 ELBO 逼近 log p(x)（第 19 章 VAE 的基石）")

    # ==================================================================
    # 4. 小结
    # ==================================================================
    section("4. 小结：这一章你亲眼看到了什么")
    print("""
  1. PCA：最大方差 = 最小重构误差（特征分解/SVD）；
  2. 压缩与白化：降维与去相关；
  3. MNIST 无监督投影 2D 自然分簇；
  4. PPCA：概率生成视角，MLE 解 = PCA 子空间；
  5. ELBO：变分下界，log p(x) 的可计算代理 —— 通向 VAE。
""")


if __name__ == "__main__":
    main()
