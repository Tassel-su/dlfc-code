# -*- coding: utf-8 -*-
"""
第 4 章：单层网络 —— 回归
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 4 章
（印刷页 111-130，PDF 页 131-150）。

本章要亲眼看到的现象：
  4.1 线性回归：
      - 基函数（多项式 / 高斯 / sigmoid）把非线性特征引入线性模型；
      - 高斯噪声假设下，极大似然估计 == 最小二乘（两者等价）；
      - 最小二乘的几何意义：把 t 投影到 Φ 的列空间；
      - 顺序学习（SGD 风格）逐步逼近批量解；
      - 正则化最小二乘（L2）抑制过拟合；
      - 多输出回归的解彼此解耦。
  4.2 决策论：平方损失下最优预测是条件期望 E[t|x]；
  4.3 偏差-方差权衡：泛化误差 = 噪声² + 偏差² + 方差，
      模型复杂度是两者的天平。

运行方式：
  C:/Python314/python.exe ch04_single_layer_regression.py
输出：
  _plots/ 下多张图 + 终端中文叙述
"""
import os
import sys

import numpy as np

# 强制 UTF-8 输出，避免 Windows 控制台打印中文报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from utils import Figure, poly_design_matrix, poly_fit_least_squares, rms_error

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


def gen_sin_data(N: int = 10, sigma: float = 0.3, seed: int = 42):
    """生成书中 1.2/4.x 的回归数据：t = sin(2πx) + N(0, σ²)。"""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, N)
    t = np.sin(2 * np.pi * x) + rng.normal(0.0, sigma, size=N)
    return x, t


def main() -> None:
    # ==================================================================
    # 4.1.1 基函数（书中 4.1.1 节）
    # ==================================================================
    section("4.1.1 基函数：把非线性特征引入线性模型（书中 4.1.1 节）")
    print("线性模型 y(x,w) = Σ_j w_j φ_j(x)：对参数 w 线性，对输入 x 可以非线性。")
    print("常用基函数：多项式、高斯、sigmoid —— 下面画出它们的形状。")

    xg = np.linspace(-1, 1, 300)

    # 高斯基函数：φ_j(x) = exp(-(x-μ_j)²/(2s²))
    mu_list = [-0.5, 0.0, 0.5]
    s = 0.15
    fig = Figure("高斯基函数 φ_j(x) = exp(-(x-μ_j)²/(2s²))", "x", "φ(x)")
    for mu in mu_list:
        phi = np.exp(-(xg - mu) ** 2 / (2 * s ** 2))
        fig.line(xg, phi, label=f"μ={mu}")
    fig.save(os.path.join(PLOTS_DIR, "fig1_gaussian_basis.png"))

    # sigmoid 基函数：φ_j(x) = σ((x-μ_j)/s)
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    fig = Figure("sigmoid 基函数 φ_j(x) = σ((x-μ_j)/s)", "x", "φ(x)")
    for mu in mu_list:
        fig.line(xg, sigmoid((xg - mu) / s), label=f"μ={mu}")
    fig.save(os.path.join(PLOTS_DIR, "fig2_sigmoid_basis.png"))
    print("  高斯基=局部响应（只在 μ 附近激活）；sigmoid 基=平滑的阶跃。")
    print("  无论基函数多复杂，模型对 w 仍是线性的 -> 闭式解依然成立。")

    # ==================================================================
    # 4.1.2 / 4.1.3 似然函数与极大似然 == 最小二乘（书中 4.1.2/4.1.3 节）
    # ==================================================================
    section("4.1.2/4.1.3 高斯噪声假设：极大似然 == 最小二乘")
    print("噪声模型：t = y(x,w) + ε，ε ~ N(0, β⁻¹)（β 是精度 = 1/方差）")
    print("=> p(t|x,w,β) = N(t | y(x,w), β⁻¹)")
    print("=> 对数似然 ln p(t|w) = -(β/2)Σ(y_n - t_n)² + 常数")
    print("=> 最大化似然 = 最小化平方和 E(w) = ½Σ(y_n - t_n)²  <- 最小二乘")

    # 数据 + 高斯基函数模型（M=4 个基函数：中心分离，条件数小，利于后续演示）
    x_train, t_train = gen_sin_data(N=20, sigma=0.3, seed=42)
    M = 4
    s = 0.2
    mus = np.linspace(0, 1, M)

    def design_matrix(x):
        """高斯基函数设计矩阵 Phi[n,j] = φ_j(x_n)。"""
        x = np.asarray(x, dtype=float)
        return np.exp(-(x[:, None] - mus[None, :]) ** 2 / (2 * s ** 2))

    Phi_raw = design_matrix(x_train)
    # 高斯基函数相邻中心重叠时，列近似共线 -> ΦᵀΦ 病态，顺序学习收敛极慢。
    # 对列做归一化（除以列标准差）：拟合曲线不变，但条件数大幅改善。
    _scale = Phi_raw.std(axis=0)
    _scale[_scale == 0] = 1.0
    Phi = Phi_raw / _scale

    # 最小二乘闭式解（书中 4.16 形式：w_ML = (ΦᵀΦ)⁻¹Φᵀt，在归一化参数下求解）
    w_ml, *_ = np.linalg.lstsq(Phi, t_train, rcond=None)

    # 数值验证：用梯度上升直接最大化对数似然，应该收敛到同一个 w
    # 对数似然对 w 的梯度：∇ = β Σ (t_n - y_n) φ_n  （令 β=1 只影响步长）
    def log_lik_grad(w):
        """对数似然梯度（β=1）。"""
        y = Phi @ w
        return Phi.T @ (t_train - y)

    w_gd = np.zeros(M)
    # 对数似然是 w 的二次函数，Hessian = -ΦᵀΦ。
    # 梯度上升的稳定步长上限是 2/λ_max，取 1.8/λ_max 保证单调收敛且尽量快。
    lr = 1.8 / np.linalg.eigvalsh(Phi.T @ Phi).max()
    for _ in range(20000):
        w_gd += lr * log_lik_grad(w_gd)          # 梯度上升（最大化对数似然）
    diff = float(np.max(np.abs(w_gd - w_ml)))
    ok = diff < 1e-6
    print(f"  梯度上升解 vs 最小二乘解：最大偏差 {diff:.2e} {'✓' if ok else '✗'}（两者等价）")
    assert ok, "极大似然与最小二乘不一致"

    # 噪声精度 β 的 MLE：β_ML⁻¹ = (1/N)Σ(t_n - y(x_n,w_ML))²
    residuals = t_train - Phi @ w_ml
    beta_ml_inv = float(np.mean(residuals ** 2))
    print(f"  噪声方差估计 β_ML⁻¹ = {beta_ml_inv:.4f}（真实 0.3²=0.09，估计稍大因自由度）")

    # ==================================================================
    # 4.1.4 最小二乘的几何（书中 4.1.4 节）
    # ==================================================================
    section("4.1.4 最小二乘的几何：把 t 投影到 Φ 的列空间")
    print("y = Φw 只能落在 Φ 的列空间 span(Φ) 里；最小二乘就是找")
    print("离 t 最近的列空间点 ŷ，残差 t - ŷ 与列空间正交。")
    # 验证正交性：Φᵀ(t - ŷ) 应等于 0（数值上 ~1e-12）
    y_hat = Phi @ w_ml
    orthogonality = Phi.T @ (t_train - y_hat)
    max_orth = float(np.abs(orthogonality).max())
    print(f"  正交性验证：Φᵀ(t - ŷ) 的最大分量 = {max_orth:.2e}（应 ~0）")
    assert max_orth < 1e-9, "投影不正交"

    # ==================================================================
    # 4.1.5 顺序学习（SGD 风格，书中 4.1.5 节）
    # ==================================================================
    section("4.1.5 顺序学习：每次用一个样本更新 w（书中 4.1.5 节）")
    print("更新规则：w ← w + η(t_n - wᵀφ_n)φ_n  （η=学习率）")
    print("这是误差 E_n = ½(t_n - wᵀφ_n)² 对 w 的梯度下降。")
    w_seq = np.zeros(M)
    eta0 = 0.1
    traj = []
    tau = 0
    # 度量用【预测差异】而非参数差异：近共线时参数有平坦方向，
    # 但预测（Φw）唯一。递减学习率 η = η0/(1+τ/1000) 满足 Robbins-Monro 条件。
    # 注意：SGD 的精确收敛需要极多轮次（最慢特征方向的限制），
    # 500 轮达到 ~0.01 量级的预测偏差已是合理演示 ——
    # "收敛速度受条件数限制"正是第 7 章优化方法的动机。
    for epoch in range(500):
        for n in range(x_train.size):
            tau += 1
            eta = eta0 / (1 + tau / 1000)
            phi_n = Phi[n]
            w_seq += eta * (t_train[n] - w_seq @ phi_n) * phi_n   # 在线更新
        traj.append(float(np.max(np.abs(Phi @ w_seq - Phi @ w_ml))))
    ok = traj[-1] < 0.05
    print(f"  500 轮顺序学习后与批量解的预测偏差 = {traj[-1]:.2e} {'✓' if ok else '✗'}")
    assert ok, "顺序学习未收敛到批量解"
    # 画收敛曲线
    fig = Figure("顺序学习：与批量最小二乘解的偏差随轮次下降", "轮次", "||w-w_ML||∞")
    fig.line(np.arange(1, 501), traj, label="偏差")
    fig.save(os.path.join(PLOTS_DIR, "fig3_sequential.png"))

    # ==================================================================
    # 4.1.6 正则化最小二乘（书中 4.1.6 节）
    # ==================================================================
    section("4.1.6 正则化最小二乘：E(w) = ½Σ(y-t)² + (λ/2)wᵀw")
    print("闭式解：w* = (ΦᵀΦ + λI)⁻¹ Φᵀ t —— λ 越大系数越小（收缩）")
    x_true = np.linspace(0, 1, 400)
    fig = Figure("高斯基模型：λ 对拟合的影响（书中 4.1.6 节风格）", "x", "t")
    fig.scatter(x_train, t_train, label="训练数据")
    for lam in (0.0, 1e-3, 1.0):
        A = Phi.T @ Phi + lam * np.eye(M)
        w_reg = np.linalg.solve(A, Phi.T @ t_train)
        fig.line(x_true, design_matrix(x_true) @ w_reg, label=f"λ={lam}")
    fig.save(os.path.join(PLOTS_DIR, "fig4_regularization.png"))
    # 展示系数收缩
    w_large_lam = np.linalg.solve(Phi.T @ Phi + 1.0 * np.eye(M), Phi.T @ t_train)
    print(f"  λ=0 时 ||w|| = {np.linalg.norm(w_ml):.3f}；λ=1 时 ||w|| = {np.linalg.norm(w_large_lam):.3f}（收缩 ✓）")

    # ==================================================================
    # 4.1.7 多输出回归（书中 4.1.7 节）
    # ==================================================================
    section("4.1.7 多输出：每个输出独立求解（书中 4.1.7 节）")
    # 两个目标：t1 = sin(2πx)+噪声，t2 = 2sin(2πx)+噪声
    rng = np.random.default_rng(7)
    x_multi = np.linspace(0, 1, 20)
    T = np.stack([
        np.sin(2 * np.pi * x_multi) + rng.normal(0, 0.3, 20),
        2 * np.sin(2 * np.pi * x_multi) + rng.normal(0, 0.3, 20),
    ], axis=1)
    Phi_m = design_matrix(x_multi)
    # 矩阵解：W = (ΦᵀΦ)⁻¹ΦᵀT —— 每个输出列独立求解，结果互不干扰
    W_ml, *_ = np.linalg.lstsq(Phi_m, T, rcond=None)
    # 验证：W 的第 k 列 == 单独用 T[:,k] 求的最小二乘解
    W_k, *_ = np.linalg.lstsq(Phi_m, T[:, 1], rcond=None)
    print(f"  多输出解 W 第 2 列 vs 单独解第 2 列：偏差 {np.max(np.abs(W_ml[:, 1] - W_k)):.2e}（解耦 ✓）")
    assert np.allclose(W_ml[:, 1], W_k)

    # ==================================================================
    # 4.2 决策论：平方损失下最优预测是 E[t|x]（书中 4.2 节）
    # ==================================================================
    section("4.2 决策论：最优回归函数是条件期望 E[t|x]（书中 4.2 节）")
    print("平方损失 L(t,y) = (y-t)² 下，使期望损失最小的预测是 h(x) = E[t|x]。")
    print("验证思路：固定 x，对同一 x 采样很多次 t，取平均 → 近似 E[t|x]。")
    x_fixed = 0.3
    h_true = np.sin(2 * np.pi * x_fixed)          # 条件期望 = 无噪声的 sin
    t_many = h_true + np.random.default_rng(1).normal(0, 0.3, 200_000)
    emp = t_many.mean()
    print(f"  x={x_fixed}：E[t|x] 理论 = {h_true:.4f}，20 万样本平均 = {emp:.4f} ✓")
    assert abs(emp - h_true) < 0.01, "条件期望验证失败"
    # 拟合模型逼近条件期望：M=6 高斯基拟合 vs E[t|x]（用大量测试点）
    x_test = np.linspace(0, 1, 500)
    model_pred = design_matrix(x_test) @ w_ml
    target_cond = np.sin(2 * np.pi * x_test)
    fig = Figure("拟合模型 vs 条件期望 E[t|x]（书中 4.2 节）", "x", "t")
    fig.line(x_test, target_cond, label="E[t|x] = sin(2πx)")
    fig.line(x_test, model_pred, label="拟合模型 y(x,w_ML)")
    fig.scatter(x_train, t_train, label="训练数据")
    fig.save(os.path.join(PLOTS_DIR, "fig5_decision_theory.png"))
    print("  好的回归模型就是在逼近 E[t|x]（条件期望）—— 这正是监督学习的真目标。")

    # ==================================================================
    # 4.3 偏差-方差权衡（书中 4.3 节）
    # ==================================================================
    section("4.3 偏差-方差权衡：误差 = 噪声² + 偏差² + 方差")
    print("对平方损失：E[(y(x;D) - h(x))²] = 噪声² + 偏差² + 方差")
    print("  偏差 = E_D[y(x;D)] - h(x)   （模型拟合能力不足）")
    print("  方差 = E_D[(y - E_D[y])²]   （对训练集的敏感度）")
    print("实验：生成 L=100 组数据集，分别拟合 M=1（欠拟合）和 M=9（过拟合），")
    print("在 400 个测试点上分解误差。")

    L, N_pts = 100, 15
    x_eval = np.linspace(0, 1, 400)
    h_eval = np.sin(2 * np.pi * x_eval)

    def decompose(M):
        """对给定阶数 M，在 L 组数据集上计算 偏差²、方差 与 噪声²。"""
        fits = np.empty((L, x_eval.size))
        for l in range(L):
            x_d, t_d = gen_sin_data(N=N_pts, sigma=0.3, seed=100 + l)
            w_l = poly_fit_least_squares(x_d, t_d, M)
            fits[l] = poly_design_matrix(x_eval, M) @ w_l
        y_avg = fits.mean(axis=0)                          # E_D[y(x;D)]
        bias2 = float(np.mean((y_avg - h_eval) ** 2))      # 偏差²
        var = float(np.mean(fits.var(axis=0)))             # 方差（对 D 平均）
        return bias2, var

    bias2_1, var_1 = decompose(1)      # 低复杂度：高偏差低方差
    bias2_9, var_9 = decompose(9)      # 高复杂度：低偏差高方差
    sigma2 = 0.3 ** 2                  # 不可约噪声
    print(f"  M=1：偏差²={bias2_1:.4f}，方差={var_1:.4f}，噪声²={sigma2}  <- 欠拟合（偏差主导）")
    print(f"  M=9：偏差²={bias2_9:.4f}，方差={var_9:.4f}，噪声²={sigma2}  <- 过拟合（方差主导）")
    total_1 = bias2_1 + var_1 + sigma2
    total_9 = bias2_9 + var_9 + sigma2
    print(f"  总期望损失：M=1 → {total_1:.4f}，M=9 → {total_9:.4f}")
    print("  结论：总误差 = 偏差²+方差²+噪声²，最优复杂度平衡两者（书中图 4.14 风格）")

    # 画偏差-方差分解对比条形
    fig = Figure("偏差-方差分解：M=1 vs M=9（书中 4.3 节）", "模型", "误差贡献")
    xs = np.array([0, 1])
    fig.line(xs, [bias2_1, bias2_9], label="偏差²")
    fig.line(xs, [var_1, var_9], label="方差")
    fig.line(xs, [sigma2, sigma2], label="噪声²")
    fig.line(xs, [total_1, total_9], label="总期望损失")
    fig.save(os.path.join(PLOTS_DIR, "fig6_bias_variance.png"))

    # 画 M=9 的多组拟合（展示高方差：曲线剧烈摆动）
    fig = Figure("M=9 在 20 组数据集上的拟合（高方差）（书中图 4.13 风格）", "x", "t")
    for l in range(20):
        x_d, t_d = gen_sin_data(N=N_pts, sigma=0.3, seed=200 + l)
        w_l = poly_fit_least_squares(x_d, t_d, 9)
        fig.line(x_eval, poly_design_matrix(x_eval, 9) @ w_l, label="")
    fig.line(x_eval, h_eval, label="真实 h(x)")
    fig.save(os.path.join(PLOTS_DIR, "fig7_high_variance.png"))

    # ==================================================================
    # 5. 小结
    # ==================================================================
    section("5. 小结：这一章你亲眼看到了什么")
    print("""
  1. 基函数让线性模型拥有非线性表达能力，但对参数仍是线性的；
  2. 高斯噪声假设下，极大似然与最小二乘完全等价；
  3. 最小二乘 = 把 t 投影到 Φ 的列空间（残差正交）；
  4. 顺序学习（SGD）逐样本更新，收敛到批量解；
  5. 正则化 = 加 λI，收缩系数、抑制过拟合；
  6. 多输出回归解耦：每个输出独立求解；
  7. 决策论：平方损失的最优预测是条件期望 E[t|x]；
  8. 偏差-方差权衡：复杂度小则偏差大，复杂度大则方差大，
     总误差 = 噪声² + 偏差² + 方差 —— 这是全书的核心主题之一。
""")


if __name__ == "__main__":
    main()
