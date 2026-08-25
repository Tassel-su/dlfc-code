# -*- coding: utf-8 -*-
"""
第 17 章：生成对抗网络（GAN）
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 17 章
（印刷页 533-552，PDF 页 553-572）。

本章要亲眼看到的现象：
  17.1 对抗训练：
      - 判别器 D 区分真假，生成器 G 骗过 D（零和博弈）；
      - 损失函数（书中 17.1.1 节）：min_G max_D V(D,G)；
      - 实际训练：交替更新 D 和 G；
  17.2 在 2D 合成数据上从零训练一个小 GAN：
      生成样本逐步逼近目标分布（4 团高斯），
      观察 D 的决策边界与 G 的生成质量变化。

运行方式：
  C:/Python314/python.exe ch17_generative_adversarial_networks.py
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
    # 17.1 对抗训练的理论（书中 17.1 节）
    # ==================================================================
    section("17.1 对抗训练：零和博弈（书中 17.1.1 节）")
    print("目标函数（书中 17.1.1 节）：")
    print("  min_G max_D  V(D,G) = E_x[ln D(x)] + E_z[ln(1 - D(G(z)))]")
    print("  D 想给真样本高分、假样本低分；G 想骗过 D")
    print("  均衡时：D(x) = p_data(x)/(p_data(x)+p_g(x))，p_g = p_data（生成完美）")

    # ==================================================================
    # 17.2 从零训练 2D GAN（书中 17.1.2 节实践）
    # ==================================================================
    section("17.2 从零训练 2D GAN（4 团高斯目标）")
    rng = np.random.default_rng(0)
    # 目标分布：4 个高斯团
    centers = np.array([[-3, -3], [3, -3], [-3, 3], [3, 3]])
    def sample_real(n):
        idx = rng.integers(0, 4, n)
        return centers[idx] + rng.normal(0, 0.6, (n, 2))

    # 生成器：噪声 z(2) -> 隐藏 16 -> tanh -> 输出 2（缩放 6 覆盖目标范围）
    # 判别器：输入 2 -> 隐藏 16 -> tanh -> sigmoid
    rngW = np.random.default_rng(1)
    G_W1 = rngW.normal(0, 0.5, (2, 16)); G_b1 = np.zeros(16)
    G_W2 = rngW.normal(0, 0.5, (16, 2)); G_b2 = np.zeros(2)
    D_W1 = rngW.normal(0, 0.5, (2, 16)); D_b1 = np.zeros(16)
    D_W2 = rngW.normal(0, 0.5, (16, 1)); D_b2 = np.zeros(1)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def generator(z, params):
        W1, b1, W2, b2 = params
        h = np.tanh(z @ W1 + b1)
        return 6.0 * np.tanh(h @ W2 + b2)          # 缩放输出到 [-6, 6]

    def discriminator(x, params):
        W1, b1, W2, b2 = params
        h = np.tanh(x @ W1 + b1)
        return sigmoid(h @ W2 + b2)

    g_params = [G_W1, G_b1, G_W2, G_b2]
    d_params = [D_W1, D_b1, D_W2, D_b2]
    lr = 0.05
    batch = 200
    gen_means = []

    for step in range(4000):
        # ========== 第一步：更新判别器 D（学会分辨真假）==========
        real = sample_real(batch)          # 真样本
        z = rng.normal(0, 1, (batch, 2))
        fake = generator(z, g_params)      # 假样本（当前生成器产出）
        d_real = discriminator(real, d_params)   # D 对真样本的输出（应接近 1）
        d_fake = discriminator(fake, d_params)   # D 对假样本的输出（应接近 0）
        # D 的损失：想让 D(real) -> 1、D(fake) -> 0
        #   -[ln D(real) + ln(1-D(fake))] 最小化
        d_loss = -float(np.mean(np.log(d_real + 1e-12) + np.log(1 - d_fake + 1e-12)))
        # D 梯度（手推链式法则）：d/dD [-ln D] = -1/D
        dd_real = -(1.0 / (d_real + 1e-12)) / batch          # dL/d(D(real))
        dd_fake = (1.0 / (1 - d_fake + 1e-12)) / batch       # dL/d(D(fake))
        def d_grad(x, dd, d_params):
            W1, b1, W2, b2 = d_params
            h = np.tanh(x @ W1 + b1)
            a = h @ W2 + b2
            sig = sigmoid(a)
            dl_a = dd * sig * (1 - sig)                       # sigmoid 导数
            dW2 = h.T @ dl_a
            db2 = dl_a.sum(axis=0)
            dh = dl_a @ W2.T * (1 - h ** 2)
            dW1 = x.T @ dh
            db1 = dh.sum(axis=0)
            return dW1, db1, dW2, db2
        gD = d_grad(real, dd_real, d_params)
        gD_f = d_grad(fake, dd_fake, d_params)
        for i in range(4):
            d_params[i] -= lr * (gD[i] + gD_f[i])

        # ---- 更新生成器 G ----
        z = rng.normal(0, 1, (batch, 2))
        fake = generator(z, g_params)
        d_fake = discriminator(fake, d_params)
        # G 损失 = -ln D(fake)（最大化骗过 D）
        g_loss = -float(np.mean(np.log(d_fake + 1e-12)))
        # G 梯度：穿过 D 再回传
        dd = -(1.0 / (d_fake + 1e-12)) / batch
        # D 对输入 x 的梯度
        W1, b1, W2, b2 = d_params
        h_d = np.tanh(fake @ W1 + b1)
        a_d = h_d @ W2 + b2
        sig_d = sigmoid(a_d)
        dl_a = dd * sig_d * (1 - sig_d)
        dx = dl_a @ W2.T * (1 - h_d ** 2) @ W1.T          # dL/d(fake)
        # G 对参数的梯度
        W1g, b1g, W2g, b2g = g_params
        h_g = np.tanh(z @ W1g + b1g)
        out_g = 6.0 * np.tanh(h_g @ W2g + b2g)
        d_out = dx * 6.0 * (1 - np.tanh(h_g @ W2g + b2g) ** 2)   # dL/d(h@W2+b2)
        gW2 = h_g.T @ d_out
        gb2 = d_out.sum(axis=0)
        dh_g = d_out @ W2g.T * (1 - h_g ** 2)
        gW1 = z.T @ dh_g
        gb1 = dh_g.sum(axis=0)
        for i, gr in enumerate([gW1, gb1, gW2, gb2]):
            g_params[i] -= lr * gr

        if step % 500 == 0:
            gen_mean = float(np.mean(np.linalg.norm(fake, axis=1)))
            gen_means.append(gen_mean)
            print(f"  step {step}: D 损失={d_loss:.3f}，G 损失={g_loss:.3f}，"
                  f"生成样本平均半径={gen_mean:.2f}（目标 ~3）")

    # 最终评估：生成样本与目标分布的匹配
    z = rng.normal(0, 1, (5000, 2))
    fake = generator(z, g_params)
    # 生成样本应分布在 4 团附近：统计"落入最近中心 1.5 以内"的比例
    dist_to_center = np.min(np.linalg.norm(fake[:, None, :] - centers[None, :, :], axis=2), axis=1)
    coverage = float(np.mean(dist_to_center < 1.5))
    print(f"  生成样本落入 4 团附近的比例 = {coverage:.3f}（目标分布覆盖率）")
    fig = Figure("GAN 训练结果：生成样本 vs 真实 4 团数据", "x1", "x2")
    real_plot = sample_real(500)
    fig.scatter(real_plot[:, 0], real_plot[:, 1], label="真实数据")
    fig.scatter(fake[::4, 0], fake[::4, 1], label="生成数据")
    fig.save(os.path.join(PLOTS_DIR, "fig1_gan_result.png"))
    print("  GAN 生成分布与目标 4 团高度重合 —— 对抗训练成功！")
    assert coverage > 0.8, "GAN 覆盖不足"

    # ==================================================================
    # 4. 小结
    # ==================================================================
    section("4. 小结：这一章你亲眼看到了什么")
    print("""
  1. GAN = 判别器与生成器的零和博弈（min-max 目标）；
  2. 交替训练：D 学分辨真假，G 学骗过 D；
  3. 均衡时生成分布 = 数据分布（纳什均衡）；
  4. 2D 演示：4000 步后生成样本覆盖 4 团目标；
  5. GAN 训练不稳定（模式坍塌/振荡）是研究热点 ——
     第 20 章的扩散模型用更稳定的目标替代了对抗训练。
""")


if __name__ == "__main__":
    main()
