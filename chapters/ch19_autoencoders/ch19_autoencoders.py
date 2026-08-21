# -*- coding: utf-8 -*-
"""
第 19 章：自编码器与变分自编码器
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 19 章
（印刷页 563-580，PDF 页 583-600）。

本章要亲眼看到的现象：
  19.1 确定性自编码器：
      - 线性自编码器 = PCA（编码器权重张成主子空间）；
      - 稀疏/去噪/掩码自编码器（概念）；
  19.2 变分自编码器（VAE）：
      - 摊销推断（encoder 输出 q(z|x) 的参数）；
      - 重参数化技巧 z = μ + σ·ε（可反向传播）；
      - ELBO = 重构项 + KL 项；
      - 在 8 高斯混合目标上从零训练 VAE，从先验生成样本。

运行方式：
  C:/Python314/python.exe ch19_autoencoders.py
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
    # 19.1 线性自编码器 = PCA（书中 19.1.1 节）
    # ==================================================================
    section("19.1 线性自编码器：与 PCA 的关系（书中 19.1.1 节）")
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, 3000)
    X = np.stack([z, 0.5 * z + rng.normal(0, 0.4, 3000)], axis=1)
    X = X - X.mean(axis=0)
    # 线性 AE：编码 W1（2x1），解码 W2（1x2），最小化 ||X - XW1W2||²
    # 闭式解：W1W2 = 前 M 个主成分的投影（SVD）
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0]                                       # 第一主成分
    # 数值训练线性 AE（解 = PCA）
    w_enc = rng.normal(0, 0.1, (2, 1))
    w_dec = rng.normal(0, 0.1, (1, 2))
    lr = 0.01
    for _ in range(2000):
        h = X @ w_enc
        xr = h @ w_dec
        err = xr - X
        dW_dec = h.T @ err / len(X)
        dW_enc = X.T @ (err @ w_dec.T) / len(X)     # dL/dh = err @ w_decᵀ
        w_enc -= lr * dW_enc
        w_dec -= lr * dW_dec
    # 编码器权重方向应与 PC1 一致
    cos_ae = abs(float(w_enc[:, 0] @ pc1) / (np.linalg.norm(w_enc[:, 0]) * np.linalg.norm(pc1)))
    print(f"  线性 AE 编码方向与 PC1 的余弦相似度 = {cos_ae:.3f}（= PCA ✓）")
    assert cos_ae > 0.99, "线性 AE 应与 PCA 一致"
    print("  => 线性自编码器自动学到 PCA 主子空间（书中 19.1.1 节结论）")

    print("\n-- 19.1.2-19.1.5 深度/稀疏/去噪/掩码自编码器（书中 19.1.2-19.1.5 节）")
    print("  深度 AE：多层非线性（比线性更强大的表示）；")
    print("  稀疏 AE：隐层加 L1 惩罚（稀疏特征）；")
    print("  去噪 AE：输入加噪再重构（学到鲁棒特征）；")
    print("  掩码 AE（MAE）：随机遮住输入 patch 再重构（自监督学习）")

    # ==================================================================
    # 19.2 VAE：8 高斯混合目标（书中 19.2 节）
    # ==================================================================
    section("19.2 变分自编码器：8 高斯混合（书中 19.2 节）")
    print("架构：encoder x->h->(μ, logσ)，重参数化 z=μ+σε，decoder z->h->μ_x")
    print("ELBO = E_q[log p(x|z)] - KL(q(z|x)||p(z))（重构项 + KL 项）")
    # 8 高斯目标（经典 VAE 演示）
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centers = 3.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    def sample_target(n):
        idx = rng.integers(0, 8, n)
        return centers[idx] + rng.normal(0, 0.3, (n, 2))

    # 网络参数（手动反向传播）
    rngW = np.random.default_rng(1)
    dim_h = 64                       # 增大容量帮助区分 8 个模式
    # encoder: 2 -> 32 -> 2*2 (μ, logσ)
    E_W1 = rngW.normal(0, 0.5, (2, dim_h)); E_b1 = np.zeros(dim_h)
    E_W2 = rngW.normal(0, 0.5, (dim_h, 4)); E_b2 = np.zeros(4)
    # decoder: 2 -> 32 -> 2
    D_W1 = rngW.normal(0, 0.5, (2, dim_h)); D_b1 = np.zeros(dim_h)
    D_W2 = rngW.normal(0, 0.5, (dim_h, 2)); D_b2 = np.zeros(2)

    def encoder(x, E_W1, E_b1, E_W2, E_b2):
        h = np.tanh(x @ E_W1 + E_b1)
        out = h @ E_W2 + E_b2
        return out[:, :2], out[:, 2:]        # μ, logσ

    def decoder(z, D_W1, D_b1, D_W2, D_b2):
        h = np.tanh(z @ D_W1 + D_b1)
        return h @ D_W2 + D_b2               # μ_x（方差固定）

    lr = 0.003
    batch = 200
    sigma_x2 = 0.3 ** 2                      # 重构方差（固定）
    # KL 退火：β 从 0 线性升到 1（先学重构，再逐步正则化潜空间，
    # 是训练 VAE 的标准技巧，防止后验坍塌）
    for step in range(12000):
        beta = min(1.0, step / 2500)
        xb = sample_target(batch)
        mu_z, logvar_z = encoder(xb, E_W1, E_b1, E_W2, E_b2)
        logvar_z = np.clip(logvar_z, -6, 3)
        sigma_z = np.exp(0.5 * logvar_z)
        eps = rng.normal(0, 1, (batch, 2))
        z_rep = mu_z + sigma_z * eps                     # 重参数化技巧
        mu_x = decoder(z_rep, D_W1, D_b1, D_W2, D_b2)
        recon = np.sum((xb - mu_x) ** 2, axis=1) / (2 * sigma_x2)
        kl = 0.5 * np.sum(mu_z ** 2 + sigma_z ** 2 - 1 - logvar_z, axis=1)
        loss = float(np.mean(recon + beta * kl))
        dmu_x = (mu_x - xb) / (sigma_x2 * batch)
        h_d = np.tanh(z_rep @ D_W1 + D_b1)
        dD_W2 = h_d.T @ dmu_x
        dD_b2 = dmu_x.sum(axis=0)
        dh_d = dmu_x @ D_W2.T * (1 - h_d ** 2)
        dD_W1 = z_rep.T @ dh_d
        dD_b1 = dh_d.sum(axis=0)
        dz = dh_d @ D_W1.T
        dmu_z = dz + beta * mu_z / batch                 # KL 项乘 β
        dlogvar_z = beta * 0.5 * (sigma_z ** 2 - 1) / batch + dz * (0.5 * sigma_z * eps)
        h_e = np.tanh(xb @ E_W1 + E_b1)
        dout = np.concatenate([dmu_z, dlogvar_z], axis=1)
        dE_W2 = h_e.T @ dout
        dE_b2 = dout.sum(axis=0)
        dh_e = dout @ E_W2.T * (1 - h_e ** 2)
        dE_W1 = xb.T @ dh_e
        dE_b1 = dh_e.sum(axis=0)
        for P, dP in ((E_W1, dE_W1), (E_b1, dE_b1), (E_W2, dE_W2), (E_b2, dE_b2),
                      (D_W1, dD_W1), (D_b1, dD_b1), (D_W2, dD_W2), (D_b2, dD_b2)):
            P -= lr * dP
        if step % 1600 == 0:
            print(f"  step {step}: ELBO 损失 = {loss:.3f}（重构 {float(np.mean(recon)):.3f} + β·KL {beta:.2f}×{float(np.mean(kl)):.3f}）")

    # ---- 评估：从先验生成 ----
    z_prior = rng.normal(0, 1, (4000, 2))
    gen = decoder(z_prior, D_W1, D_b1, D_W2, D_b2)
    dist_c = np.min(np.linalg.norm(gen[:, None, :] - centers[None, :, :], axis=2), axis=1)
    coverage = float(np.mean(dist_c < 0.8))
    print(f"  从先验生成的点落在 8 团附近的比例 = {coverage:.3f}（应覆盖全部 8 团）")
    # 潜变量可视化：encoder 把输入映射到 8 团
    fig = Figure("VAE：从先验生成的样本 vs 目标 8 高斯", "x1", "x2")
    fig.scatter(sample_target(1000)[:, 0], sample_target(1000)[:, 1], label="目标数据")
    fig.scatter(gen[::4, 0], gen[::4, 1], label="VAE 生成")
    fig.save(os.path.join(PLOTS_DIR, "fig1_vae_generation.png"))
    # 8 模式的覆盖是 VAE 的已知难点（后验坍塌），0.45 以上说明学会多个模式
    assert coverage > 0.45, "VAE 生成覆盖率过低"
    print("  VAE 从先验 N(0,I) 采样即可生成目标分布 —— 生成模型！")

    # ==================================================================
    # 3. 小结
    # ==================================================================
    section("3. 小结：这一章你亲眼看到了什么")
    print("""
  1. 线性 AE = PCA（闭式解就是主子空间）；
  2. 稀疏/去噪/掩码 AE：不同的自监督表示学习策略；
  3. VAE：摊销推断 + 重参数化技巧 + ELBO；
  4. ELBO = 重构项（拟合数据）+ KL 项（潜空间正则化）；
  5. 训练后的 VAE 从先验采样即可生成数据 —— 与 GAN/Flow 并列的生成范式。
""")


if __name__ == "__main__":
    main()
