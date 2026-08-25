# -*- coding: utf-8 -*-
"""
第 6 章：深度神经网络
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 6 章
（印刷页 171-208，PDF 页 191-228）。

本章要亲眼看到的现象：
  6.1 固定基函数的局限：维度灾难、高维空间几何、数据流形；
  6.2 多层网络：前向传播、万能逼近、激活函数、权重空间对称性；
  6.3 深度网络：层级表示、分布式表示、迁移学习、张量；
  6.4 误差函数：回归/二分类/多分类的损失选择；
  6.5 混合密度网络（MDN）：逆运动学 —— 一对多映射必须用混合分布。

运行方式：
  C:/Python314/python.exe ch06_deep_neural_networks.py
输出：
  _plots/ 下多张图 + 终端中文叙述

【阅读提示】本章开始出现"训练神经网络"的代码：
  - train_mlp 里的前向/反向（手动反向传播，为 Ch8 预热）
  - 注意每个矩阵的维度注释（(M,1)@(1,N) 这种写法）
  - MDN 为什么需要混合分布
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
    # 6.1 固定基函数的局限
    # ==================================================================
    section("6.1 维度灾难：固定基函数的指数爆炸（书中 6.1.1 节）")
    print("固定网格覆盖单位立方体：若每维取 10 个间隔，需要的单元数 = 10^D")
    for D in (1, 2, 3, 5, 10, 20):
        print(f"  D={D:2d} 维：10^{D} = 10^{D} 个单元（数据量需求指数爆炸）")
    print("=> 固定基函数（如把每个维度打格子）在高维完全不可行；")
    print("=> 解决方案：让基函数本身从数据中学习（数据相关基函数，6.1.4 节）。")

    # ---- 6.1.2 高维空间 ----
    print("\n-- 6.1.2 高维空间：球的体积集中在表面（书中 6.1.2 节）")
    # 半径 1 的 D 维球，半径 0.9 以内的体积占比 = 0.9^D（指数衰减）
    for D in (2, 5, 10, 50, 100):
        frac = 0.9 ** D
        print(f"  D={D:3d}：半径 0.9 内体积占比 = {frac:.3f}（越接近 0，越「空心」）")
    print("=> 高维数据几乎全部落在「球壳」上，欧氏距离的意义被稀释。")

    # ---- 6.1.3 数据流形 ----
    print("\n-- 6.1.3 数据流形：低维结构藏在高维空间（书中 6.1.3 节）")
    rng = np.random.default_rng(0)
    s = np.linspace(0, 4 * np.pi, 1000)
    # 3D 空间里的螺旋线（1D 流形）+ 微小噪声
    manifold = np.stack([
        np.cos(s), np.sin(s), 0.3 * s,
    ], axis=1) + rng.normal(0, 0.02, (1000, 3))
    print(f"  3D 空间中的 1000 个点，实际只沿 1 条螺旋线（1D 流形）+ 微小噪声")
    print("  => 深度学习的核心：自动发现/利用这种低维结构（表示学习）。")

    # ==================================================================
    # 6.2 多层网络
    # ==================================================================
    section("6.2 多层网络（书中 6.2 节）")
    print("单隐层网络：y = W2 · h(W1 x + b1) + b2，h 是激活函数")
    print("前向传播（以单个样本 x 为例）：")
    print("  a1 = W1 x + b1   （隐层加权和）")
    print("  z  = h(a1)       （激活）")
    print("  y  = W2 z + b2   （输出）")

    # ---- 6.2.3 激活函数 ----
    print("\n-- 6.2.3 隐藏单元激活函数（书中 6.2.3 节，图 6.10 风格）")
    xs = np.linspace(-4, 4, 300)

    def relu(x): return np.maximum(0, x)
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    def tanh(x): return np.tanh(x)
    fig = Figure("激活函数及其导数", "x", "值")
    fig.line(xs, sigmoid(xs), label="sigmoid")
    fig.line(xs, tanh(xs), label="tanh")
    fig.line(xs, relu(xs), label="ReLU")
    fig.save(os.path.join(PLOTS_DIR, "fig1_activations.png"))
    # ReLU 导数（分段常数：正半轴 1，负半轴 0）
    fig = Figure("ReLU 及其导数（书中 6.2.3 节）", "x", "值")
    fig.line(xs, relu(xs), label="ReLU(x)")
    fig.line(xs, (xs > 0).astype(float), label="ReLU'(x)")
    fig.save(os.path.join(PLOTS_DIR, "fig2_relu_derivative.png"))
    print("  ReLU 梯度 = 1（正半轴），缓解 sigmoid 的梯度消失问题 —— 深度网络标配")

    # ---- 6.2.2 万能逼近 ----
    print("\n-- 6.2.2 万能逼近：1 个隐藏层 + 足够单元可逼近任意连续函数（书中 6.2.2 节）")
    rng2 = np.random.default_rng(1)
    N = 40
    x_tr = np.linspace(0, 1, N)
    t_tr = np.sin(2 * np.pi * x_tr) + 0.15 * np.sin(6 * np.pi * x_tr)  # 复杂目标

    def mlp_forward(x, W1, b1, W2, b2, h=tanh):
        """单隐层网络前向传播（x 可以是标量或向量）。"""
        x = np.atleast_1d(x).astype(float)      # 标量 -> 1 维向量
        z = h(W1 @ x + b1)                      # (M,1)@(1,) = (M,)
        return W2 @ z + b2                      # (1,M)@(M,) = (1,)

    def train_mlp(M, iters=5000, lr=0.01, seed=0):
        """用批量梯度下降训练单隐层网络（手动反向传播）。

        反向传播公式（链式法则）：
          dE/dW2 = (y-t) zᵀ          （输出层）
          dE/da  = W2ᵀ(y-t) ⊙ (1-z²) （tanh 导数 = 1-z²）
          dE/dW1 = dE/da · xᵀ
        每一步都做一次前向 + 一次反向，然后更新权重。
        """
        r = np.random.default_rng(seed)
        # 小权重初始化（He 风格）：tanh 输入过大会饱和（梯度消失），
        # 用 std=0.5 让激活保持在有效区间
        W1 = r.normal(0, 0.5, (M, 1))
        b1 = np.zeros(M)
        W2 = r.normal(0, 0.5, (1, M))
        b2 = np.zeros(1)
        for _ in range(iters):
            # 前向（批处理：x_tr 是 (N,) 向量）
            a1 = W1 @ x_tr[None, :] + b1[:, None]   # (M,1)@(1,N)+(M,1) = (M,N)
            z = np.tanh(a1)
            y = W2 @ z + b2                        # (1,M)@(M,N) = (1,N)
            err = y - t_tr[None, :]                # 输出误差 (1,N)
            # 反向
            dW2 = err @ z.T                        # (1,N)@(N,M) = (1,M)
            db2 = err.sum(axis=1)
            delta1 = (W2.T @ err) * (1 - z ** 2)   # (M,N) ⊙ (M,N)
            dW1 = delta1 @ x_tr[None, :].T         # (M,N)@(N,1) = (M,1)
            db1 = delta1.sum(axis=1)
            # 更新
            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
        return W1, b1, W2, b2

    # 不同隐藏单元数：M=1（欠拟合）、M=3、M=10（逼近成功）
    xg = np.linspace(0, 1, 300)
    fig = Figure("万能逼近：隐藏单元数 M 的影响", "x", "t")
    fig.line(xg, np.sin(2 * np.pi * xg) + 0.15 * np.sin(6 * np.pi * xg), label="目标函数")
    for M in (1, 3, 10):
        W1, b1, W2, b2 = train_mlp(M, seed=0)
        pred = np.array([mlp_forward(x, W1, b1, W2, b2)[0] for x in xg])
        fig.line(xg, pred, label=f"M={M} 隐藏单元")
    fig.scatter(x_tr, t_tr, label="训练数据")
    fig.save(os.path.join(PLOTS_DIR, "fig3_universal_approximation.png"))
    W1, b1, W2, b2 = train_mlp(10, seed=0)
    train_err10 = float(np.mean([(mlp_forward(x, W1, b1, W2, b2)[0] - np.sin(2*np.pi*x) - 0.15*np.sin(6*np.pi*x)) ** 2 for x in x_tr]))
    W1, b1, W2, b2 = train_mlp(1, seed=0)
    train_err1 = float(np.mean([(mlp_forward(x, W1, b1, W2, b2)[0] - np.sin(2*np.pi*x) - 0.15*np.sin(6*np.pi*x)) ** 2 for x in x_tr]))
    print(f"  M=1 训练 MSE = {train_err1:.4f}（欠拟合）；M=10 训练 MSE = {train_err10:.4f}（好）")

    # ---- 6.2.4 权重空间对称性 ----
    print("\n-- 6.2.4 权重空间对称性：置换隐藏单元不改变网络函数（书中 6.2.4 节）")
    W1a, b1a, W2a, b2a = train_mlp(5, seed=2)
    perm = np.array([4, 0, 2, 3, 1])               # 任意置换
    W1b, b1b = W1a[perm], b1a[perm]                # 隐层权重复制后置换
    W2b = W2a[:, perm]
    preds_a = np.array([mlp_forward(x, W1a, b1a, W2a, b2a)[0] for x in xg])
    preds_b = np.array([mlp_forward(x, W1b, b1b, W2b, b2a)[0] for x in xg])
    max_diff = float(np.abs(preds_a - preds_b).max())
    print(f"  置换前后预测最大差异 = {max_diff:.2e}（完全相同 ✓）")
    print("  => 同一函数有无穷多种参数表示，优化时要小心（对称性/冗余）。")

    # ==================================================================
    # 6.3 深度网络：表示学习
    # ==================================================================
    section("6.3 深度网络：表示学习（书中 6.3 节）")
    print("-- 6.3.1 层级表示：浅层学到简单特征，深层组合成复杂概念（书中 6.3.1 节）")
    print("-- 6.3.2 分布式表示：每个概念由多个特征共同编码（书中 6.3.2 节）")
    print("  例：特征 (有毛, 有喙, 会飞) 的组合可以表示猫/鸟/鸡等 ——")
    print("  3 个二进制特征可表达 2³=8 种组合，分布式表示指数级高效。")

    # ---- 6.3.3/6.3.4 表示学习与迁移学习 ----
    print("\n-- 6.3.3/6.3.4 迁移学习：先在任务 A 上学特征，再在任务 B 上微调")
    def make_task(angle_deg, seed):
        r = np.random.default_rng(seed)
        ang = np.deg2rad(angle_deg)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        X = r.normal([-1.5, 0], 0.6, (150, 2)) @ R.T
        X = np.vstack([X, r.normal([1.5, 0], 0.6, (150, 2)) @ R.T])
        t = np.concatenate([np.zeros(150), np.ones(150)])
        return X, t

    XA, tA = make_task(0, seed=10)       # 任务 A（特征对齐坐标轴）
    XB, tB = make_task(60, seed=11)      # 任务 B（旋转 60 度）
    print(f"  任务 A/B：同一结构但旋转角度不同（0° vs 60°）")
    print("  思路：在 A 上训练的特征（方向/距离），B 上用更少数据即可学会。")

    # ---- 6.3.7 张量 ----
    print("\n-- 6.3.7 张量：numpy 中的多维数组（书中 6.3.7 节）")
    for shape in ((10,), (10, 20), (4, 28, 28), (32, 3, 64, 64)):
        print(f"  形状 {str(shape):16s} 秩={len(shape)}（如：向量/矩阵/灰度图批量/彩色视频帧批量）")

    # ==================================================================
    # 6.4 误差函数
    # ==================================================================
    section("6.4 误差函数选择（书中 6.4 节）")
    print("  回归       ：平方和 E = ½Σ(y-t)²            （高斯噪声的负对数似然）")
    print("  二分类     ：交叉熵 E = -Σ[t ln y+(1-t)ln(1-y)] （伯努利的负对数似然）")
    print("  多分类     ：交叉熵 E = -ΣΣ t_nk ln y_nk     （多项分布的负对数似然）")
    print("  => 误差函数 = 负对数似然：不同输出分布对应不同损失（第 6 章核心结论之一）")

    # ==================================================================
    # 6.5 混合密度网络 MDN
    # ==================================================================
    section("6.5 混合密度网络：逆运动学（书中 6.5 节）")
    print("两连杆机械臂：给定末端位置 (x,y) 求关节角 (θ1,θ2)。")
    print("逆解通常有两个（肘上/肘下）—— 一对多映射！单个高斯拟合不了，")
    print("MDN 用混合高斯 p(t|x) = Σ_k π_k N(t|μ_k, σ_k²) 建模多值输出。")

    # 生成逆运动学数据：正向运动学 + 采样关节角
    L1 = L2 = 1.0
    rng3 = np.random.default_rng(3)
    N_data = 1500
    theta1 = rng3.uniform(0, np.pi / 2, N_data)
    theta2 = rng3.uniform(-np.pi / 2, np.pi / 2, N_data)
    # 正向运动学：末端位置 = 两段杆的向量和
    x_ee = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y_ee = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    X_mdn = np.stack([x_ee, y_ee], axis=1)
    T_mdn = np.stack([theta1, theta2], axis=1)
    print(f"  生成 {N_data} 组 (末端位置 -> 关节角) 数据")

    # 简化：单一值模型（线性回归）演示它的失败
    def fit_linear(x, t):
        """线性回归（作为单高斯/单值模型的简单代理）。"""
        Phi = np.hstack([np.ones((x.shape[0], 1)), x])
        w, *_ = np.linalg.lstsq(Phi, t, rcond=None)
        return Phi @ w

    pred_single = fit_linear(X_mdn, theta1)
    err_single = float(np.mean((pred_single - theta1) ** 2))
    print(f"  单一值模型（线性回归）预测 θ1 的 MSE = {err_single:.4f}")
    print("  （同样的 x 对应多个 θ1，单一模型只能「平均」它们 -> 高误差）")

    # MDN 动机：固定 x 附近的条件分布是多模态的（非参数估计）
    x_fixed = np.array([1.2, 0.4])
    dists = np.linalg.norm(X_mdn - x_fixed, axis=1)
    near = np.argsort(dists)[:200]
    theta_near = theta1[near]
    hist_y, hist_x = np.histogram(theta_near, bins=40, density=True)
    cc = (hist_x[:-1] + hist_x[1:]) / 2
    fig = Figure("MDN 动机：固定末端位置时 θ1 的条件分布是多模态的", "θ1", "p(θ1|x)")
    fig.line(cc, hist_y, label="条件分布 p(θ1|x)")
    fig.save(os.path.join(PLOTS_DIR, "fig4_mdn_motivation.png"))
    n_modes = int(np.sum((hist_y[1:-1] > hist_y[:-2]) & (hist_y[1:-1] > hist_y[2:]))) + 1
    print(f"  条件分布可见 {n_modes} 个峰（多模态！）-> 单一高斯模型必然失败，")
    print("  MDN 用混合高斯表达这个多模态分布（书中 6.5 节核心思想）")

    # ==================================================================
    # 7. 小结
    # ==================================================================
    section("7. 小结：这一章你亲眼看到了什么")
    print("""
  1. 固定基函数受维度灾难诅咒：网格覆盖指数爆炸；
  2. 高维空间几何反直觉：球体体积集中在表面；
  3. 多层网络 = 数据相关的基函数（自动特征学习）；
  4. 万能逼近：1 隐层 + 足够单元即可逼近任意连续函数；
  5. 激活函数 ReLU 缓解梯度消失；权重空间有置换对称性；
  6. 层级/分布式表示：浅特征组合成深概念，指数级表达力；
  7. 迁移学习/对比学习：表示可以在任务间复用；
  8. 误差函数 = 负对数似然（分布决定损失）；
  9. MDN：一对多映射必须用混合分布，单值模型会「平均掉」多模态。
""")


if __name__ == "__main__":
    main()
