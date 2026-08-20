# -*- coding: utf-8 -*-
"""
第 7 章：梯度下降
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 7 章
（印刷页 209-232，PDF 页 229-252）。

本章要亲眼看到的现象：
  7.1 误差曲面：局部二次近似、条件数决定收敛速度；
  7.2 梯度下降家族：批量 GD / 随机 SGD / 小批量 mini-batch；
      参数初始化（全零是陷阱）；
  7.3 收敛加速：动量法（momentum）、学习率调度、RMSProp、Adam；
  7.4 归一化：数据标准化改善条件数、批归一化 / 层归一化。

运行方式：
  C:/Python314/python.exe ch07_gradient_descent.py
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
    # 7.1 误差曲面：局部二次近似与条件数（书中 7.1 节）
    # ==================================================================
    section("7.1 误差曲面：E(w) ≈ E(w*) + ½(w-w*)ᵀH(w-w*)（书中 7.1.1 节）")
    print("最小值附近误差曲面可用二次型近似，Hessian H 的特征值决定「谷」的形状。")
    print("条件数 = λ_max/λ_min：越大，谷越狭长，普通梯度下降越慢。")

    # 二维二次型：E(w) = ½(w1² + c*w2²)，c 控制条件数
    def quad(c):
        H = np.diag([1.0, c])
        return H

    for c in (1.0, 10.0, 100.0):
        print(f"  条件数 λ_max/λ_min = {c:.0f}：谷宽比 {1:.0f}:{1/np.sqrt(c):.3f}（越狭长越难优化）")

    # 画误差曲面等高线（条件数 10）
    ws1 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(ws1, ws1)
    E = 0.5 * (W1 ** 2 + 10 * W2 ** 2)
    fig = Figure("二次误差曲面 E(w) = ½(w1² + 10·w2²)", "w1", "w2")
    # 等高线用散点着色近似（低值=深色不可用，这里画几条等高线轨迹）
    for level in (1, 4, 16, 64):
        theta = np.linspace(0, 2 * np.pi, 200)
        r = np.sqrt(2 * level)
        fig.line(r * np.cos(theta), r / np.sqrt(10) * np.sin(theta), label=f"E={level}")
    fig.save(os.path.join(PLOTS_DIR, "fig1_error_surface.png"))
    print("  等高线是椭圆：长轴沿 w1（平坦方向），短轴沿 w2（陡峭方向）")

    # ==================================================================
    # 7.2 梯度下降家族（书中 7.2 节）
    # ==================================================================
    section("7.2 批量 / 随机 / 小批量梯度下降（书中 7.2 节）")
    print("目标：最小化一个简单的二次函数 E(w) = ½Σ(wᵀφ_n - t_n)²（线性回归问题）")

    # 构造一个小型线性回归问题
    rng = np.random.default_rng(0)
    N = 100
    X = rng.normal(0, 1, (N, 2))
    t_true = X @ np.array([1.5, -1.0]) + rng.normal(0, 0.1, N)

    def full_grad(w):
        """全批量梯度。"""
        return X.T @ (X @ w - t_true) / N

    def sgd_step(w, idx):
        """单个样本梯度。"""
        return X[idx] * (X[idx] @ w - t_true[idx])

    def run_optimizer(grad_fn, n_steps, lr, momentum=0.0, sgd=False, batch_size=1, seed=0):
        """通用优化循环，返回每步误差。"""
        w = np.zeros(2)
        v = np.zeros(2)
        errs = []
        r = np.random.default_rng(seed)
        perm = r.permutation(N) if sgd else None
        for step in range(n_steps):
            if sgd:
                idx = perm[step % N]
                g = grad_fn(w, idx)
            else:
                g = grad_fn(w)
            v = momentum * v - lr * g            # 动量更新
            w = w + v
            errs.append(float(np.sum((X @ w - t_true) ** 2) / 2))
        return w, errs

    w_opt, *_ = np.linalg.lstsq(X, t_true, rcond=None)

    # ---- 批量 GD vs SGD vs 小批量 ----
    n_steps = 200
    lr = 0.1
    w_batch, err_batch = run_optimizer(full_grad, n_steps, lr)
    w_sgd, err_sgd = run_optimizer(lambda w, i: sgd_step(w, i), n_steps, lr * 20, sgd=True)
    print(f"  批量 GD  200 步后误差 = {err_batch[-1]:.4f}（平滑下降）")
    print(f"  SGD      200 步后误差 = {err_sgd[-1]:.4f}（波动大但前期快）")

    # 画收敛曲线
    fig = Figure("批量 GD vs SGD 收敛（书中 7.2 节）", "迭代步", "误差")
    fig.line(np.arange(1, n_steps + 1), err_batch, label="批量 GD (lr=0.1)")
    fig.line(np.arange(1, n_steps + 1), err_sgd, label="SGD (lr=2.0)")
    fig.save(os.path.join(PLOTS_DIR, "fig2_batch_vs_sgd.png"))
    print("  SGD 单步计算便宜、能逃离局部极小、天然处理大数据 —— 深度网络标配")

    # ---- 7.2.5 参数初始化：全零是陷阱 ----
    print("\n-- 7.2.5 参数初始化：全零初始化的对称性问题（书中 7.2.5 节）")
    print("  两层网络若 W1=0：所有隐藏单元收到相同输入、梯度相同 -> 永远对称")
    print("  => 用小的随机初始化打破对称性（如 N(0, 0.01)）")

    # ==================================================================
    # 7.3 收敛加速：动量 / 调度 / RMSProp / Adam（书中 7.3 节）
    # ==================================================================
    section("7.3 动量 / 学习率调度 / RMSProp / Adam（书中 7.3 节）")

    # 用一个病态二次型展示：E = ½(w1² + 100 w2²)，条件数 100
    def ill_grad(w):
        return np.array([w[0], 100.0 * w[1]])

    def run_quad(opt, n_steps, **kw):
        w = np.array([2.0, 0.4])
        v = np.zeros(2)
        m = np.zeros(2); s = np.zeros(2)
        errs = []
        lr = kw.get("lr", 0.05)
        beta = kw.get("beta", 0.9)
        beta2 = kw.get("beta2", 0.999)
        eps = 1e-8
        for t in range(1, n_steps + 1):
            g = ill_grad(w)
            if opt == "gd":
                w = w - lr * g
            elif opt == "momentum":
                v = beta * v - lr * g
                w = w + v
            elif opt == "rmsprop":
                s = 0.9 * s + 0.1 * g ** 2
                w = w - lr * g / (np.sqrt(s) + eps)
            elif opt == "adam":
                m = beta * m + (1 - beta) * g
                s = beta2 * s + (1 - beta2) * g ** 2
                mh = m / (1 - beta ** t)
                sh = s / (1 - beta2 ** t)
                w = w - lr * mh / (np.sqrt(sh) + eps)
            errs.append(float(0.5 * (w[0] ** 2 + 100 * w[1] ** 2)))
        return w, errs

    n_steps = 500
    results = {}
    # 普通 GD/动量受最大特征值限制（lr 必须 < 2/λ_max = 0.02），
    # 自适应方法（RMSProp/Adam）步长归一化，可用大得多的 lr。
    lr_by_opt = {"gd": 0.01, "momentum": 0.01, "rmsprop": 0.1, "adam": 0.1}
    for name in ("gd", "momentum", "rmsprop", "adam"):
        w_end, errs = run_quad(name, n_steps, lr=lr_by_opt[name])
        results[name] = errs
        print(f"  {name:9s} 500 步后误差 = {errs[-1]:.3e}（w*=[0,0]）")
    fig = Figure("病态二次型（条件数 100）上各优化器收敛对比（书中 7.3 节）", "迭代步", "误差")
    for name, errs in results.items():
        fig.line(np.arange(1, n_steps + 1), errs, label=name)
    fig.save(os.path.join(PLOTS_DIR, "fig3_optimizers.png"))
    print("  Adam 自适应每维步长（除以 √s），在病态问题上远快于普通 GD")

    # ---- 7.3.2 学习率调度 ----
    print("\n-- 7.3.2 学习率调度：先大后小（书中 7.3.2 节）")
    # 加梯度噪声模拟 SGD：固定 lr 会在最优解附近徘徊（噪声地板），
    # 衰减学习率把步长逐步缩小，才能继续压到地板以下。
    rng_noise = np.random.default_rng(5)
    w = np.array([2.0, 0.4])
    errs_const = []
    for _ in range(3000):
        w = w - 0.015 * (ill_grad(w) + rng_noise.normal(0, 0.5, 2))
        errs_const.append(float(0.5 * (w[0] ** 2 + 100 * w[1] ** 2)))
    w = np.array([2.0, 0.4])
    errs_decay = []
    for t in range(1, 3001):
        lr_t = 0.015 * 0.9 ** (t / 300)
        w = w - lr_t * (ill_grad(w) + rng_noise.normal(0, 0.5, 2))
        errs_decay.append(float(0.5 * (w[0] ** 2 + 100 * w[1] ** 2)))
    print(f"  SGD 固定 lr 3000 步误差 = {errs_const[-1]:.3e}（停在噪声地板附近）")
    print(f"  SGD 指数衰减 3000 步误差 = {errs_decay[-1]:.3e}（衰减后继续下降 ✓）")
    fig = Figure("学习率调度：固定 vs 指数衰减", "迭代步", "误差")
    fig.line(np.arange(1, 3001), errs_const, label="固定 lr=0.015")
    fig.line(np.arange(1, 3001), errs_decay, label="指数衰减")
    fig.save(os.path.join(PLOTS_DIR, "fig4_lr_schedule.png"))

    # ==================================================================
    # 7.4 归一化（书中 7.4 节）
    # ==================================================================
    section("7.4 归一化：数据标准化改善条件数（书中 7.4 节）")
    # 构造尺度差异巨大的特征：x1 量级 1，x2 量级 1000
    rng4 = np.random.default_rng(4)
    X_scale = np.hstack([rng4.normal(0, 1, (200, 1)), rng4.normal(0, 1000, (200, 1))])
    t_scale = X_scale @ np.array([1.0, 0.01]) + rng4.normal(0, 0.1, 200)

    def lin_grad(w):
        return X_scale.T @ (X_scale @ w - t_scale) / 200

    w = np.zeros(2)
    # 未标准化：最大特征值 ~2e8，稳定步长上限 2/λ_max ~ 1e-8。
    # 用各自的最优稳定步长对比，500 步后看谁离最优解更近。
    lam_raw = np.linalg.eigvalsh(X_scale.T @ X_scale / 200).max()
    lr_raw = 1.8 / lam_raw
    errs_raw = []
    for _ in range(500):
        w = w - lr_raw * lin_grad(w)
        errs_raw.append(float(np.sum((X_scale @ w - t_scale) ** 2) / 2))
    # 标准化后
    mu = X_scale.mean(axis=0); sd = X_scale.std(axis=0)
    X_norm = (X_scale - mu) / sd
    t_norm = t_scale - t_scale.mean()
    w = np.zeros(2)
    lam_norm = np.linalg.eigvalsh(X_norm.T @ X_norm / 200).max()
    lr_norm = 1.8 / lam_norm
    errs_norm = []
    for _ in range(500):
        g = X_norm.T @ (X_norm @ w - t_norm) / 200
        w = w - lr_norm * g
        errs_norm.append(float(np.sum((X_norm @ w - t_norm) ** 2) / 2))
    opt_raw = float(np.sum((X_scale @ np.linalg.lstsq(X_scale, t_scale, rcond=None)[0] - t_scale) ** 2) / 2)
    print(f"  未标准化：500 步误差 = {errs_raw[-1]:.2e}（最优解误差 {opt_raw:.2f}，还差得远）")
    print(f"  标准化后：500 步误差 = {errs_norm[-1]:.2e}（已达最优 ~{opt_raw:.2f} ✓）")
    print(f"  原因：未标准化时最大特征值 ~{lam_raw:.0e}，被迫用步长 ~{lr_raw:.1e}，")
    print(f"        平坦方向（小特征值）500 步根本走不完；标准化后条件数回到 O(1)。")
    fig = Figure("数据标准化对收敛的影响（书中 7.4.1 节）", "迭代步", "误差")
    fig.line(np.arange(1, 501), errs_raw, label="未标准化")
    fig.line(np.arange(1, 501), errs_norm, label="标准化")
    fig.save(os.path.join(PLOTS_DIR, "fig5_normalization.png"))

    print("\n-- 7.4.2/7.4.3 批归一化与层归一化（书中 7.4.2/7.4.3 节）")
    print("  批归一化：对每个 mini-batch 的激活做标准化（减均值除标准差 + 缩放平移），")
    print("  让每层输入分布稳定，可用更大的学习率；")
    print("  层归一化：对单个样本的整层激活做标准化，不依赖 batch 大小，")
    print("  适用于 RNN/Transformer。")

    # ==================================================================
    # 5. 小结
    # ==================================================================
    section("5. 小结：这一章你亲眼看到了什么")
    print("""
  1. 误差曲面在最小值附近是二次型，条件数决定收敛难度；
  2. 批量 GD 平滑但慢；SGD 快且能逃离局部极小；小批量是折中；
  3. 参数初始化要打破对称性（全零是陷阱）；
  4. 动量加速病态问题；学习率调度先大后小；RMSProp/Adam 自适应步长；
  5. 数据标准化把条件数拉回 1，收敛速度天壤之别；
  6. 批/层归一化稳定深层网络的激活分布。
""")


if __name__ == "__main__":
    main()
