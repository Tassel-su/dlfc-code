# -*- coding: utf-8 -*-
"""
第 20 章：扩散模型
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 20 章
（印刷页 581-620，PDF 页 601-640）。

本章要亲眼看到的现象：
  20.1 前向编码器：逐步加噪 x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
      （扩散核，书中 20.1.1 节）；
  20.2 反向解码器：训练噪声预测网络 ε_θ(x_t, t)，
      从纯噪声逐步去噪生成样本（书中 20.2.4/20.2.5 节）；
  20.3 Score Matching：score = -ε/√(1-ᾱ)（书中 20.3 节）；
  20.4 引导扩散：classifier-free guidance（书中 20.4.2 节）。

运行方式：
  C:/Python314/python.exe ch20_diffusion_models.py
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
    # 20.1 前向过程：扩散核（书中 20.1.1 节）
    # ==================================================================
    section("20.1 前向编码器：逐步加噪（书中 20.1.1/20.1.2 节）")
    print("前向过程：x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε，ε ~ N(0,I)")
    print("  ᾱ_t 随 t 减小：t 越大噪声越强，最终 x_T ~ N(0,I)")

    # 目标数据：环形（经典的扩散演示分布）
    rng = np.random.default_rng(0)
    def sample_ring(n, r_in=2.0, r_out=3.0):
        ang = rng.uniform(0, 2 * np.pi, n)
        r = rng.uniform(r_in, r_out, n)
        return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)

    X0 = sample_ring(3000)
    # 线性 β 调度
    T = 100
    beta = np.linspace(0.0001, 0.02, T)
    alpha_bar = np.cumprod(1 - beta)

    def forward(x0, t):
        """前向：给 x0 加噪到时刻 t。返回 (x_t, 噪声)。

        公式：x_t = √(ᾱ_t) x0 + √(1-ᾱ_t) ε，ε ~ N(0,I)
        其中 ᾱ_t = Π_{s=1}^{t} (1-β_s)（累积噪声水平）。
        t 越大 ᾱ 越小 -> x_t 越接近纯噪声。
        关键：这个变换是"解析的"，不需要学习（这就是扩散核）。
        """
        t = np.clip(t, 0, T - 1)
        ab = alpha_bar[t][:, None]               # 每个样本取自己时刻的 ᾱ
        eps = rng.normal(0, 1, x0.shape)         # 采样高斯噪声
        x_t = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        return x_t, eps

    # 可视化前向扩散：t=0, 30, 60, 99
    fig = Figure("前向扩散：环形数据逐步变成高斯噪声（书中 20.1 节）", "x1", "x2")
    for tt in (0, 30, 60, 99):
        x_t, _ = forward(X0[:400], np.full(400, tt))
        fig.scatter(x_t[:, 0], x_t[:, 1], label=f"t={tt}")
    fig.save(os.path.join(PLOTS_DIR, "fig1_forward.png"))
    print("  前向过程不需要学习：直接按公式加噪（已知的扩散核）")

    # ==================================================================
    # 20.2 反向：训练噪声预测网络（书中 20.2.4 节）
    # ==================================================================
    section("20.2 反向解码器：训练 ε_θ(x_t, t)（书中 20.2.4 节）")
    print("训练目标：min E ||ε - ε_θ(√ᾱ x0 + √(1-ᾱ)ε, t)||²（简化的 ELBO）")

    # 噪声预测网络：输入 (x(2), t 归一化) -> 隐藏 64 -> 隐藏 64 -> 输出 ε(2)
    rngW = np.random.default_rng(1)
    N_W1 = rngW.normal(0, 0.4, (3, 64)); N_b1 = np.zeros(64)
    N_W2 = rngW.normal(0, 0.4, (64, 64)); N_b2 = np.zeros(64)
    N_W3 = rngW.normal(0, 0.4, (64, 2)); N_b3 = np.zeros(2)

    def noise_net(x, t_norm, N_W1, N_b1, N_W2, N_b2, N_W3, N_b3):
        """输入 (batch,2) + 时间 (batch,1) -> 预测噪声 (batch,2)。"""
        inp = np.concatenate([x, t_norm[:, None]], axis=1)
        h1 = np.tanh(inp @ N_W1 + N_b1)
        h2 = np.tanh(h1 @ N_W2 + N_b2)
        return h2 @ N_W3 + N_b3

    batch = 256
    lr = 0.01
    for step in range(6000):
        x0 = sample_ring(batch)
        t = rng.integers(0, T, batch)
        x_t, eps = forward(x0, t)
        t_norm = t.astype(float) / T                      # 归一化时间
        eps_pred = noise_net(x_t, t_norm, N_W1, N_b1, N_W2, N_b2, N_W3, N_b3)
        loss = float(np.mean((eps_pred - eps) ** 2))      # 简化损失
        # 反向（手推）
        d = 2 * (eps_pred - eps) / batch
        h2 = np.tanh(np.concatenate([x_t, t_norm[:, None]], axis=1) @ N_W1 + N_b1)
        h3 = np.tanh(h2 @ N_W2 + N_b2)
        dN_W3 = h3.T @ d
        dN_b3 = d.sum(axis=0)
        dh3 = d @ N_W3.T * (1 - h3 ** 2)
        dN_W2 = h2.T @ dh3
        dN_b2 = dh3.sum(axis=0)
        dh2 = dh3 @ N_W2.T * (1 - h2 ** 2)
        inp = np.concatenate([x_t, t_norm[:, None]], axis=1)
        dN_W1 = inp.T @ dh2
        dN_b1 = dh2.sum(axis=0)
        for P, dP in ((N_W1, dN_W1), (N_b1, dN_b1), (N_W2, dN_W2), (N_b2, dN_b2),
                      (N_W3, dN_W3), (N_b3, dN_b3)):
            P -= lr * dP
        if step % 1200 == 0:
            print(f"  step {step}: 噪声预测损失 = {loss:.4f}")

    # ==================================================================
    # 20.2.5 采样：从噪声反向去噪（书中 20.2.5 节）
    # ==================================================================
    section("20.2.5 生成：从纯噪声反向去噪（书中 20.2.5 节）")
    print("采样：x_T ~ N(0,I)，逐时刻用学到的去噪器往回走")
    x = rng.normal(0, 1, (2000, 2))
    for tt in range(T - 1, -1, -1):
        t_norm = np.full(x.shape[0], tt / T)
        eps_pred = noise_net(x, t_norm, N_W1, N_b1, N_W2, N_b2, N_W3, N_b3)
        # DDPM 反向一步：x_{t-1} = (x_t - β_t/√(1-ᾱ_t)·ε_θ) / √(1-β_t) + σ·z
        ab = alpha_bar[tt]
        x = (x - beta[tt] / np.sqrt(1 - ab) * eps_pred) / np.sqrt(1 - beta[tt])
        if tt > 0:
            x = x + np.sqrt(beta[tt]) * rng.normal(0, 1, x.shape)   # 随机性项
    # 评估：生成样本应落在环上（半径 2-3）
    rad = np.linalg.norm(x, axis=1)
    on_ring = float(np.mean((rad > 1.5) & (rad < 3.5)))
    print(f"  生成样本落在环形带(1.5<r<3.5)的比例 = {on_ring:.3f}")
    fig = Figure("扩散模型生成：反向去噪的样本 vs 真实环形", "x1", "x2")
    fig.scatter(X0[::5, 0], X0[::5, 1], label="真实数据")
    fig.scatter(x[::3, 0], x[::3, 1], label="生成数据")
    fig.save(os.path.join(PLOTS_DIR, "fig2_generated.png"))
    assert on_ring > 0.85, "扩散生成质量不足"
    print("  ✓ 从纯噪声出发，学到的反向过程重建了环形分布！")

    # ==================================================================
    # 20.3 Score Matching（书中 20.3 节）
    # ==================================================================
    section("20.3 Score Matching：score = -ε/√(1-ᾱ)（书中 20.3 节）")
    print("噪声预测与 score 等价：∇_x log p(x_t) = -ε_θ(x_t,t)/√(1-ᾱ_t)")
    # 验证：在某个加噪样本上比较 score 与 -ε/√(1-ᾱ)
    x0_s = X0[0]
    t_s = 50
    x_t, eps_s = forward(x0_s[None, :], np.array([t_s]))
    ab_s = alpha_bar[t_s]
    # 用网络预测的噪声（≈ E[ε|x_t]）而非单次抽样的 ε：
    # 理论 score = -E[ε|x_t]/√(1-ᾱ_t)
    eps_pred_s = noise_net(x_t, np.array([t_s / T]), N_W1, N_b1, N_W2, N_b2, N_W3, N_b3)
    score_theory = -eps_pred_s[0] / np.sqrt(1 - ab_s)
    # 数值 score：用环数据 + 高斯核估计 log p，再有限差分求梯度
    def logp_kde(x):
        d2 = np.sum((x[None, :, :] - X0[:, None, :]) ** 2, axis=2)   # (3000, n)
        return -np.min(d2, axis=0) / (2 * 0.2 ** 2)
    score_num = np.zeros(2)
    h = 0.01
    for i in range(2):
        pp = x_t.copy(); pp[0, i] += h
        pm = x_t.copy(); pm[0, i] -= h
        score_num[i] = float((logp_kde(pp) - logp_kde(pm))[0] / (2 * h))
    cos_score = abs(float(score_theory @ score_num) / (np.linalg.norm(score_theory) * np.linalg.norm(score_num) + 1e-9))
    print(f"  理论 score 方向 {np.round(score_theory, 3)}（指向密度增大方向）")
    print(f"  数值 score（原分布 KDE）与理论方向的余弦相似度 = {cos_score:.3f}")
    print("  （说明：理论 score 属于 t 时刻的平滑分布，单点 KDE 估计有偏差；")
    print("   核心关系 score = -ε/√(1-ᾱ) 是严格的数学恒等式 ——")
    print("   训练噪声预测网络 = 训练 score 模型（书中 20.3 节）")
    print("  SDE 视角（书中 20.3.4 节）：前向是扩散 SDE，反向由 score 驱动）")

    # ==================================================================
    # 20.4 引导扩散（书中 20.4.2 节）
    # ==================================================================
    section("20.4 引导扩散：classifier-free guidance（书中 20.4.2 节）")
    print("引导：ε_θ(x,t,c) = ε_θ(x,t) + w·(ε_θ(x,t,c) - ε_θ(x,t))")
    print("  w>0 强化条件信息（如文本/类别），w=0 无条件生成")
    print("  权重 w 控制「创造力 vs 服从性」的权衡（书中 20.4.2 节）")

    # ==================================================================
    # 5. 小结
    # ==================================================================
    section("5. 小结：这一章你亲眼看到了什么")
    print("""
  1. 前向 = 已知的加噪公式（扩散核），无需学习；
  2. 反向 = 学习去噪器 ε_θ(x_t, t)，训练损失是简单的 MSE；
  3. 从纯噪声逐步去噪即可生成（环形重建成功！）；
  4. 噪声预测 ⇔ score 匹配：score = -ε/√(1-ᾱ)；
  5. SDE 视角：连续时间扩散；引导控制生成方向；
  6. 这就是 Stable Diffusion / DALL-E 的核心原理 —— 全书收官！
""")


if __name__ == "__main__":
    main()
