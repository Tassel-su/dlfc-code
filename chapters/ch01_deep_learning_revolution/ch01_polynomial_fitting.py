# -*- coding: utf-8 -*-
"""
第 1 章：深度学习革命 —— 多项式曲线拟合
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 1 章 1.2-1.3 节
（PDF 第 4-14 页）。

本章要亲眼看到的现象：
  1. 用多项式 y(x, w) = Σ_j w_j x^j 拟合带噪的 sin(2πx) 数据；
  2. 模型阶数 M 太小 -> 欠拟合；M 太大 -> 过拟合（训练误差小、测试误差大）；
  3. 训练/测试 RMS 误差曲线（书中图 1.5）呈现经典的 U 形；
  4. L2 正则化抑制过拟合：M=9 + ln λ = -18 时系数被收缩（书中图 1.7）；
  5. 留一法 (LOO) 模型选择：不靠"看测试集"也能选出 M=3（书中 1.3 节）。

运行方式：
  C:/Python314/python.exe ch01_polynomial_fitting.py
输出：
  _plots/ 下 5 张图 + 终端中文叙述
"""
import os
import sys

import numpy as np

# 把仓库根目录加入路径，便于 import utils
# 脚本位于 repo/chapters/chXX/ 下，需要向上 3 层才到仓库根
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from utils import (Figure, check_gradient, gen_sin_data, poly_design_matrix,
                   poly_eval, poly_fit_least_squares, rms_error)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    """打印小节标题分隔线。"""
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. 数据生成：t = sin(2πx) + ε，ε ~ N(0, σ²)          （书中 1.2 节）
    # ------------------------------------------------------------------
    section("1. 数据生成：带噪的 sin(2πx)（书中 1.2 节，PDF p.4）")
    N = 10                 # 训练样本数（与书中一致）
    sigma = 0.3            # 噪声标准差（与书中一致）
    x_train, t_train = gen_sin_data(N=N, sigma=sigma, seed=42)
    print(f"训练数据：{N} 个点，x ∈ [0,1] 均匀分布，t = sin(2πx) + N(0, {sigma}²)")
    print(f"x = {np.round(x_train, 3)}")
    print(f"t = {np.round(t_train, 3)}")

    # 高密度的"真实函数 + 干净测试集"，用来评估泛化误差（书中用大量测试点）
    x_true = np.linspace(0, 1, 500)
    t_true = np.sin(2 * np.pi * x_true)
    rng = np.random.default_rng(123)
    x_test = np.linspace(0, 1, 1000)
    t_test = np.sin(2 * np.pi * x_test) + rng.normal(0.0, sigma, size=x_test.size)

    fig = Figure("第 1 章：训练数据（10 个带噪点）与真实函数 sin(2πx)", "x", "t")
    fig.line(x_true, t_true, label="真实函数 sin(2πx)")
    fig.scatter(x_train, t_train, label="训练数据")
    fig.save(os.path.join(PLOTS_DIR, "fig1_sin_data.png"))

    # ------------------------------------------------------------------
    # 2. 最小二乘正规方程 + 两种验证（闭式解对照 / 数值梯度检验）
    # ------------------------------------------------------------------
    section("2. 多项式基函数与最小二乘正规方程（书中 1.2 节）")
    print("模型：y(x, w) = Σ_j w_j x^j，设计矩阵 Phi[n,j] = x_n^j")
    print("误差函数：E(w) = ½ Σ_n (y(x_n,w) - t_n)²  ->  闭式解 w* = (ΦᵀΦ)⁻¹ Φᵀ t")

    M = 9  # 先取一个高阶级数做验证
    # 验证①：M=3 时，正规方程闭式解 与 SVD 的 lstsq 解应当一致（验证公式本身正确）
    M_small = 3
    w_solve = poly_fit_least_squares(x_train, t_train, M_small, method="solve")
    w_lstsq3 = poly_fit_least_squares(x_train, t_train, M_small, method="lstsq")
    diff_small = np.max(np.abs(w_solve - w_lstsq3))
    assert diff_small < 1e-9, f"M=3 正规方程与 lstsq 不一致：{diff_small}"
    print(f"验证① M=3 时正规方程解 vs lstsq：最大偏差 {diff_small:.2e} ✓（公式正确）")

    # 验证①'：M=9 时 ΦᵀΦ 病态，正规方程直接解会数值崩溃 —— 这正是要用 SVD/lstsq 的原因
    Phi9 = poly_design_matrix(x_train, M)
    cond9 = np.linalg.cond(Phi9.T @ Phi9)
    w_solve9 = poly_fit_least_squares(x_train, t_train, M, method="solve")
    w_lstsq9 = poly_fit_least_squares(x_train, t_train, M, method="lstsq")
    diff9 = np.max(np.abs(w_solve9 - w_lstsq9))
    print(f"验证①' M=9 时 cond(ΦᵀΦ) ≈ {cond9:.2e}（病态！），"
          f"正规方程 vs lstsq 偏差 {diff9:.1f} —— 所以代码用 lstsq")
    print(f"       （书本公式 w*=(ΦᵀΦ)⁻¹Φᵀt 在理论上正确，数值上要交给 SVD）")

    # 验证②：误差函数 E(w) 的解析梯度 vs 中心有限差分（书中 8.5 节方法的提前预热）
    #   解析梯度：∇E = Φᵀ (Φw - t)
    def error(w):
        return 0.5 * float(np.sum((Phi9 @ w - t_train) ** 2))

    def grad_error(w):
        return Phi9.T @ (Phi9 @ w - t_train)

    w0 = np.linspace(-2, 2, M + 1)
    max_diff, ok = check_gradient(error, grad_error, w0)
    assert ok, "梯度检验未通过"
    print(f"验证② E(w) 解析梯度 vs 有限差分：最大误差 {max_diff:.2e} ✓")

    # ------------------------------------------------------------------
    # 3. 不同阶数 M = 0..9 的拟合：欠拟合 vs 过拟合（书中图 1.4 / 1.5）
    # ------------------------------------------------------------------
    section("3. 不同阶数的拟合效果：欠拟合 vs 过拟合（书中图 1.4/1.5）")
    M_list = list(range(0, 10))
    rms_train, rms_test = [], []
    fits = {}  # M -> 系数
    for m in M_list:
        w_m = poly_fit_least_squares(x_train, t_train, m)
        fits[m] = w_m
        rms_train.append(rms_error(t_train, poly_eval(x_train, w_m)))
        rms_test.append(rms_error(t_test, poly_eval(x_test, w_m)))

    print(" M | 训练 RMS | 测试 RMS   <- E_RMS = sqrt( 2E(w*)/N )，书中公式")
    print("---+" + "-" * 30)
    for m, (r_tr, r_te) in zip(M_list, zip(rms_train, rms_test)):
        print(f"{m:2d} |  {r_tr:.4f}  |  {r_te:.4f}")

    # 图：四条典型拟合曲线（对应书中图 1.4 的 M=0,1,3,9）
    fig = Figure("不同阶数多项式拟合（书中图 1.4 风格）", "x", "t")
    fig.line(x_true, t_true, label="真实函数")
    for m in (0, 1, 3, 9):
        fig.line(x_true, poly_eval(x_true, fits[m]), label=f"M={m}")
    fig.scatter(x_train, t_train, label="训练数据")
    fig.save(os.path.join(PLOTS_DIR, "fig2_poly_fits.png"))

    # 图：RMS 随 M 变化（书中图 1.5：测试误差 U 形曲线）
    fig = Figure("训练/测试 RMS 随阶数 M 变化（书中图 1.5）", "M", "E_RMS")
    fig.line(M_list, rms_train, label="训练 RMS")
    fig.line(M_list, rms_test, label="测试 RMS")
    fig.save(os.path.join(PLOTS_DIR, "fig3_rms_vs_M.png"))

    best_m_test = int(np.argmin(rms_test))
    print(f"测试集上最优 M = {best_m_test}（训练集上 RMS 随 M 单调下降，测试集呈 U 形 = 过拟合信号）")

    # ------------------------------------------------------------------
    # 4. L2 正则化：M=9 时用 λ 收缩系数（书中 1.2 节末，图 1.7）
    # ------------------------------------------------------------------
    section("4. L2 正则化抑制过拟合（书中 1.2 节，PDF p.9-11）")
    print("正则化误差：Ẽ(w) = ½ Σ_n (y(x_n,w)-t_n)² + (λ/2)||w||²")
    print("闭式解：w* = (ΦᵀΦ + λI)⁻¹ Φᵀ t")
    print("书中用 M=9：ln λ = -∞（无正则）时振荡剧烈；ln λ = -18 时被拉回平滑")

    fig = Figure("M=9 时 L2 正则化的效果（书中图 1.7 风格）", "x", "t")
    fig.line(x_true, t_true, label="真实函数")
    for ln_lam in (-np.inf, -18.0, 0.0):
        lam = 0.0 if np.isneginf(ln_lam) else np.exp(ln_lam)
        w_reg = poly_fit_least_squares(x_train, t_train, M, lam=lam)
        fig.line(x_true, poly_eval(x_true, w_reg),
                 label="无正则 (lnλ=-∞)" if lam == 0 else f"ln λ = {ln_lam:.0f}")
    fig.scatter(x_train, t_train, label="训练数据")
    fig.save(os.path.join(PLOTS_DIR, "fig4_regularization.png"))

    print("M=9 的系数 w_j（感受正则化的收缩效应）：")
    for ln_lam in (-np.inf, -18.0, 0.0):
        lam = 0.0 if np.isneginf(ln_lam) else np.exp(ln_lam)
        w_reg = poly_fit_least_squares(x_train, t_train, M, lam=lam)
        label = "无正则" if lam == 0 else f"lnλ={ln_lam:.0f}"
        print(f"  {label:10s}: {np.round(w_reg, 3)}")

    # ------------------------------------------------------------------
    # 5. 留一法 (LOO) 模型选择（书中 1.3 节，PDF p.11-14）
    # ------------------------------------------------------------------
    section("5. 留一法模型选择：不碰测试集选出 M=3（书中 1.3 节）")
    print("LOO：对每个样本 n，用除 n 外的 N-1 个点训练，再预测第 n 个点，")
    print("累计平方误差 E_LOO(M) = Σ_n (t_n - y^(n)(x_n))²。")

    loo_errors = []
    for m in M_list:
        err_sum = 0.0
        for n in range(N):
            idx = np.arange(N) != n
            w_m = poly_fit_least_squares(x_train[idx], t_train[idx], m)
            err_sum += (t_train[n] - poly_eval(np.array([x_train[n]]), w_m)[0]) ** 2
        loo_errors.append(err_sum)
    best_m_loo = int(np.argmin(loo_errors))

    fig = Figure("留一法误差随 M 变化（书中图 1.10 风格）", "M", "E_LOO")
    fig.line(M_list, loo_errors, label="留一法误差")
    fig.save(os.path.join(PLOTS_DIR, "fig5_loo.png"))

    print(" M | LOO 误差")
    print("---+" + "-" * 14)
    for m, e in zip(M_list, loo_errors):
        print(f"{m:2d} | {e:.4f}")
    print(f"LOO 选出的最优 M = {best_m_loo}（书中结论：M=3 最优）")

    # ------------------------------------------------------------------
    # 6. 本章小结
    # ------------------------------------------------------------------
    section("6. 小结：这一章你亲眼看到了什么")
    print("""
  1. 阶数 M 控制模型容量：M 小欠拟合（偏差大），M 大过拟合（方差大）；
  2. 训练 RMS 随 M 单调下降，测试 RMS 是 U 形 —— 泛化误差才是真标准；
  3. L2 正则化通过收缩系数（尤其高阶项）把 M=9 的振荡拉回平滑；
  4. LOO 无需独立验证集即可做模型选择，选出 M=3 —— 与书一致；
  5. 这些概念（容量/正则化/模型选择）是全书的骨架，后续章节反复使用。
""")


if __name__ == "__main__":
    main()
