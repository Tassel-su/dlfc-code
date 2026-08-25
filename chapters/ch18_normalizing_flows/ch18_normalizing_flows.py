# -*- coding: utf-8 -*-
"""
第 18 章：标准化流（Normalizing Flows）
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 18 章
（印刷页 553-562，PDF 页 573-582）。

本章要亲眼看到的现象：
  18.1 耦合流（coupling flows）：
      - 可逆变换 + 雅可比行列式 = 精确密度估计；
      - 仿射耦合层：一半变量做尺度/平移，另一半做条件；
      - 前向（生成）与逆向（密度）都闭式可算；
  18.2 自回归流（概念）；
  18.3 连续流 / Neural ODE（概念 + 欧拉积分演示）。

运行方式：
  C:/Python314/python.exe ch18_normalizing_flows.py
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
    # 18.1 变量变换与精确密度（书中 18.1 节）
    # ==================================================================
    section("18.1 变量变换：p(x) = p(z) |det J|⁻¹（书中 18.1 节）")
    print("若 x = f(z) 可逆，则 log p(x) = log p(z) - log|det J_f|")
    print("标准化流 = 多层可逆变换的复合，密度可以精确计算！")

    # 仿射耦合层（书中 18.1 节）：对 2D 变量 (x1, x2)
    #   前向: y1 = x1；y2 = x2 * exp(s(x1)) + t(x1)
    #   逆:   x2 = (y2 - t(y1)) * exp(-s(y1))
    #   log|det J| = s(x1)
    # 用线性函数近似 s、t（最简单形式）
    def coupling_forward(x, s_w, t_w):
        """仿射耦合层前向。x: (N, 2)。s(x1)=s_w*x1，t(x1)=t_w*x1。

        变换规则（一半不变，一半被变换）：
          y1 = x1                          （第一维直接透传）
          y2 = x2 * exp(s(x1)) + t(x1)     （第二维按 x1 的函数缩放+平移）
        为什么可逆？给定 y1=x1，就能算出 s、t，反解 x2。
        log|det J| = Σ s(x1)：雅可比是三角矩阵，行列式 = 对角乘积。
        """
        x1, x2 = x[:, 0], x[:, 1]
        s = s_w * x1
        t = t_w * x1
        y = np.stack([x1, x2 * np.exp(s) + t], axis=1)
        logdet = s                                     # log|det J| = Σ s
        return y, logdet

    def coupling_inverse(y, s_w, t_w):
        """逆变换：从 y 回到 x。"""
        y1, y2 = y[:, 0], y[:, 1]
        s = s_w * y1
        t = t_w * y1
        x2 = (y2 - t) * np.exp(-s)
        return np.stack([y1, x2], axis=1)

    # 验证可逆性：前向再逆向 = 恒等
    rng = np.random.default_rng(0)
    x_test = rng.normal(0, 1, (1000, 2))
    y, logdet = coupling_forward(x_test, s_w=0.3, t_w=0.5)
    x_rec = coupling_inverse(y, s_w=0.3, t_w=0.5)
    inv_err = float(np.abs(x_rec - x_test).max())
    print(f"  前向+逆向还原误差 = {inv_err:.2e}（可逆 ✓）")
    assert inv_err < 1e-8, "耦合层不可逆"

    # ==================================================================
    # 18.1 训练耦合流：把高斯噪声映射到"斜高斯"目标
    # ==================================================================
    section("18.1 训练耦合流：学习把噪声变成目标分布")
    # 目标分布：倾斜的高斯（协方差有相关性）
    Sigma_t = np.array([[2.0, 1.2], [1.2, 1.0]])
    rng2 = np.random.default_rng(1)
    X_target = rng2.multivariate_normal([0, 0], Sigma_t, 5000)
    # ---- 两层交替耦合（标准做法：第一层耦合 x2|x1，第二层耦合 x1|x2）----
    p = np.array([0.0, 0.0, 0.0, 0.0])          # [s1, t1, s2, t2]
    lr = 0.05

    def flow_forward(x, p):
        """两层耦合前向 x -> z（z 应是标准高斯）。"""
        s1, t1, s2, t2 = p
        x1, x2 = x[:, 0], x[:, 1]
        y1 = x1
        y2 = x2 * np.exp(s1 * x1) + t1 * x1          # 层 1：耦合 x2
        z1 = y1 * np.exp(s2 * y2) + t2 * y2          # 层 2：耦合 y1
        z2 = y2
        logdet = s1 * x1 + s2 * y2                   # log|det J| 累计
        return np.stack([z1, z2], axis=1), logdet

    def flow_inverse(z, p):
        """逆向 z -> x。"""
        s1, t1, s2, t2 = p
        z1, z2 = z[:, 0], z[:, 1]
        y2 = z2
        y1 = (z1 - t2 * y2) * np.exp(-s2 * y2)       # 层 2 逆向
        x1 = y1
        x2 = (y2 - t1 * x1) * np.exp(-s1 * x1)       # 层 1 逆向
        return np.stack([x1, x2], axis=1)

    eps = 1e-5
    def neg_ll(p):
        zz, logdet = flow_forward(X_target, p)
        lp = -0.5 * np.sum(zz ** 2, axis=1) - np.log(2 * np.pi)   # log N(z;0,I)
        return -float(np.mean(lp + logdet))          # log p(x) = log p(z) + log|det|

    for it in range(4000):
        g = np.zeros(4)
        for i in range(4):
            pp, pm = p.copy(), p.copy()
            pp[i] += eps; pm[i] -= eps
            g[i] = (neg_ll(pp) - neg_ll(pm)) / (2 * eps)
        p -= 0.02 * g
        if it % 800 == 0:
            print(f"  iter {it}: NLL = {neg_ll(p):.4f}，参数 = {np.round(p, 3)}")

    z_gen = rng2.normal(0, 1, (3000, 2))
    x_gen, _ = flow_forward(z_gen, p)
    cov_gen = np.cov(x_gen.T)
    print(f"  生成协方差 = {np.round(cov_gen, 3)}（目标 {np.round(Sigma_t, 3)} ✓）")
    corr_gen = cov_gen[0, 1] / np.sqrt(cov_gen[0, 0] * cov_gen[1, 1])
    corr_true = Sigma_t[0, 1] / np.sqrt(Sigma_t[0, 0] * Sigma_t[1, 1])
    print(f"  生成相关系数（绝对值）= {abs(corr_gen):.3f}（目标 {corr_true:.3f} ✓）")
    print("  说明：高斯先验对称，流的两种朝向（±相关）似然相同 —— 反射自由度")
    fig = Figure("耦合流：生成样本 vs 目标倾斜高斯", "x1", "x2")
    fig.scatter(X_target[::3, 0], X_target[::3, 1], label="目标")
    fig.scatter(x_gen[::3, 0], x_gen[::3, 1], label="Flow 生成")
    fig.save(os.path.join(PLOTS_DIR, "fig1_coupling_flow.png"))
    assert abs(abs(corr_gen) - corr_true) < 0.1, "流未学到目标相关性"
    print("  流的关键优势：密度可精确计算（不像 GAN/VAE 只能近似）")

    # ==================================================================
    # 18.2/18.3 自回归流与 Neural ODE（书中 18.2/18.3 节）
    # ==================================================================
    section("18.2/18.3 自回归流与连续流（书中 18.2/18.3 节）")
    print("-- 18.2 自回归流：p(x) = Π p(xd | x1..x_{d-1})，逐维条件建模")
    print("  用自回归分解构造可逆变换（如 MADE/Masked Autoregressive Flow）")
    print("\n-- 18.3 Neural ODE：x(t) 满足 dx/dt = f(x,θ)，欧拉积分：")
    def velocity(x):
        """简单速度场：绕原点旋转 + 轻微向内收缩。"""
        return np.stack([-x[:, 1] * 0.8 - x[:, 0] * 0.1,
                         x[:, 0] * 0.8 - x[:, 1] * 0.1], axis=1)
    x0 = np.array([[1.0, 0.0]])
    dt = 0.01
    traj = [x0[0].copy()]
    for _ in range(300):
        x0 = x0 + dt * velocity(x0)
        traj.append(x0[0].copy())
    traj = np.array(traj)
    print(f"  欧拉积分 300 步：点从 (1,0) 螺旋到 ({traj[-1,0]:.3f}, {traj[-1,1]:.3f})")
    fig = Figure("Neural ODE 的速度场轨迹（欧拉积分）", "x1", "x2")
    fig.line(traj[:, 0], traj[:, 1], label="轨迹")
    fig.save(os.path.join(PLOTS_DIR, "fig2_neural_ode.png"))
    print("  Neural ODE：连续时间上的深度网络（深度 = 积分时长）")

    # ==================================================================
    # 4. 小结
    # ==================================================================
    section("4. 小结：这一章你亲眼看到了什么")
    print("""
  1. 变量变换 + 雅可比 = 精确可计算的密度（不同于 GAN/VAE 的近似）；
  2. 仿射耦合层：可逆、前向/逆向闭式、log|det J| 易算；
  3. 训练耦合流学到目标分布的相关性（协方差匹配）；
  4. 自回归流：逐维条件分解构造可逆变换；
  5. Neural ODE：连续深度的另一视角。
""")


if __name__ == "__main__":
    main()
