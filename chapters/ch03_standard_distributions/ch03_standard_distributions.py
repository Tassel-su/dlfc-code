# -*- coding: utf-8 -*-
"""
第 3 章：标准分布
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 3 章
（印刷页 65-110，PDF 页 85-130）。

本章要亲眼看到的现象：
  3.1 离散分布：伯努利 / 二项 / 多项 —— pmf、均值方差、极大似然估计；
  3.2 多元高斯：几何形状（特征分解 -> 椭球）、条件分布 / 边际分布公式、
      高斯贝叶斯定理（共轭）、MLE、顺序估计（Robbins-Monro）、高斯混合；
  3.3 周期变量：von Mises 分布（环形数据）；
  3.4 指数族：一般形式 p(x|η)=h(x)g(η)exp(ηᵀu(x))，
      伯努利和高斯都是特例，u(x) 即充分统计量；
  3.5 非参数方法：直方图（箱宽效应）、核密度估计（带宽效应）、
      k 近邻密度估计。

运行方式：
  C:/Python314/python.exe ch03_standard_distributions.py
输出：
  _plots/ 下多张图 + 终端中文叙述
"""
import os
import sys

# 强制 UTF-8 输出，避免 Windows 控制台 cp1252 打印中文报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from utils import (Figure, bernoulli_pmf, binomial_pmf, entropy,
                   gaussian_pdf, multivariate_gaussian_pdf, multinomial_pmf)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ==================================================================
    # 3.1 离散分布（书中 3.1 节）
    # ==================================================================
    section("3.1 离散分布：伯努利 / 二项 / 多项（书中 3.1 节）")

    # ---- 3.1.1 伯努利分布 ----
    print("-- 伯努利 Bern(x|μ) = μ^x (1-μ)^(1-x)，x∈{0,1}（书中 3.1.1 节）")
    mu = 0.6
    print(f"  p(1|μ={mu}) = {bernoulli_pmf(1, mu):.3f}，p(0|μ) = {bernoulli_pmf(0, mu):.3f}")
    # 均值 E[x] = μ，方差 var[x] = μ(1-μ) —— 用大量样本验证
    samples = np.random.default_rng(0).random(200_000) < mu
    print(f"  样本均值 {samples.mean():.4f}（理论 μ={mu}），"
          f"样本方差 {samples.var():.4f}（理论 μ(1-μ)={mu*(1-mu):.4f}）")
    # MLE：μ_ML = (1/N) Σ x_n（即正面比例）
    mu_ml = float(samples.mean())
    print(f"  MLE: μ_ML = 样本均值 = {mu_ml:.4f}（N 越大越接近真实 μ）")

    # ---- 3.1.2 二项分布 ----
    print("\n-- 二项 Bin(m|N,μ) = C(N,m) μ^m (1-μ)^(N-m)（书中 3.1.2 节）")
    N_bin, mu_bin = 10, 0.3
    ms = np.arange(0, N_bin + 1)
    pmfs = np.array([binomial_pmf(m, N_bin, mu_bin) for m in ms])
    print(f"  N={N_bin}, μ={mu_bin}：最大概率出现在 m={int(ms[np.argmax(pmfs)])}（约 Nμ={N_bin*mu_bin}）")
    # 归一化验证：所有 m 的 pmf 之和 = 1
    assert abs(pmfs.sum() - 1.0) < 1e-9, "二项分布未归一化"
    # 均值 = Nμ，方差 = Nμ(1-μ)
    print(f"  验证：Σpmf={pmfs.sum():.6f}；E[m]=Nμ={N_bin*mu_bin}，Var=Nμ(1-μ)={N_bin*mu_bin*(1-mu_bin):.4f}")
    # 画 pmf 条形（用散点+线近似）
    fig = Figure("二项分布 Bin(m|10, 0.3) 的 pmf", "m", "p(m)")
    fig.line(ms, pmfs, label="pmf")
    fig.scatter(ms, pmfs, label="取值点")
    fig.save(os.path.join(PLOTS_DIR, "fig1_binomial.png"))

    # ---- 3.1.3 多项分布 ----
    print("\n-- 多项 Mult(m1..mK|μ1..μK)（书中 3.1.3 节）")
    probs = np.array([0.3, 0.5, 0.2])
    counts = np.array([3, 5, 2])
    p_multi = multinomial_pmf(counts, probs)
    print(f"  抽取 10 次，得 (3,5,2) 的概率 = {p_multi:.5f}")
    # MLE：μ_k = m_k / N
    mu_mle = counts / counts.sum()
    print(f"  MLE: μ_ML = m_k/N = {np.round(mu_mle, 3)}（理论 μ={probs}）")

    # ==================================================================
    # 3.2 多元高斯（书中 3.2 节）
    # ==================================================================
    section("3.2 多元高斯：几何 / 条件 / 边际 / 贝叶斯 / MLE / 混合")

    # ---- 3.2.1 几何：协方差特征分解 -> 椭球 ----
    print("-- 3.2.1 几何：Σ 的特征分解决定椭球的轴长与方向")
    Sigma = np.array([[2.0, 0.6], [0.6, 1.0]])
    eigvals, eigvecs = np.linalg.eigh(Sigma)          # 对称矩阵特征分解
    print(f"  特征值 λ = {np.round(eigvals, 3)}，特征向量方向 = {np.round(eigvecs, 3)}")
    print(f"  椭球半轴长 = √λ = {np.round(np.sqrt(eigvals), 3)}（沿特征向量方向）")

    # ---- 3.2.2 矩（moments）：均值与协方差 ----
    print("\n-- 3.2.2 矩：E[x]=μ，cov[x]=Σ（用样本验证）")
    rng = np.random.default_rng(1)
    mu2 = np.array([1.0, -2.0])
    xs = rng.multivariate_normal(mu2, Sigma, size=100_000)
    print(f"  样本均值 {np.round(xs.mean(axis=0), 3)}（理论 {mu2}）")
    print(f"  样本协方差 {np.round(np.cov(xs.T), 3)}（理论 {Sigma}）")

    # ---- 3.2.3 局限：单高斯不能表示多峰 ----
    print("\n-- 3.2.3 局限：单个高斯只能有一个峰（书中图 3.6 讨论）")
    # 生成双峰数据，用单个高斯拟合，直观看到"单峰拟合双峰"的失败
    comp1 = rng.multivariate_normal([-2, 0], [[0.5, 0], [0, 0.5]], 50_000)
    comp2 = rng.multivariate_normal([2, 0], [[0.5, 0], [0, 0.5]], 50_000)
    bimodal = np.vstack([comp1, comp2])
    mu_fit = bimodal.mean(axis=0)                     # 单高斯的 MLE 均值
    cov_fit = np.cov(bimodal.T)
    print(f"  双峰数据拟合出的单高斯：均值 {np.round(mu_fit,2)}（夹在两峰中间），")
    print(f"  协方差 {np.round(np.diag(cov_fit),2)}（被两峰拉大）—— 这就是单高斯的局限")

    # ---- 3.2.4/3.2.5 条件分布与边际分布（书中 3.2.4/3.2.5 节）----
    print("\n-- 3.2.4/3.2.5 条件与边际（分块矩阵公式 + 数值验证）")
    # 分块：x = (x_a, x_b)，a 是 1 维，b 是 1 维
    mu_ab = np.array([1.0, 2.0])
    Sig_aa = np.array([[1.5]])
    Sig_ab = np.array([[0.8]])
    Sig_ba = np.array([[0.8]])
    Sig_bb = np.array([[2.0]])
    # 条件分布公式：p(x_a|x_b) = N( μ_a + Σ_ab Σ_bb⁻¹ (x_b-μ_b), Σ_aa - Σ_ab Σ_bb⁻¹ Σ_ba )
    x_b_val = 1.0
    # 1x1 情形直接标量运算（等价于矩阵公式 Σ_ab Σ_bb⁻¹ (x_b-μ_b)）
    mu_cond = mu_ab[0] + (Sig_ab[0, 0] / Sig_bb[0, 0]) * (x_b_val - mu_ab[1])
    var_cond = float(Sig_aa[0, 0] - Sig_ab[0, 0] * Sig_ba[0, 0] / Sig_bb[0, 0])
    print(f"  给定 x_b={x_b_val}：p(x_a|x_b) 均值 = {mu_cond:.3f}，方差 = {var_cond:.3f}")
    # 数值验证：从联合分布采样，再筛出 x_b 接近给定值的样本，比较其 x_a 分布
    Sigma_full = np.array([[1.5, 0.8], [0.8, 2.0]])
    samples2 = rng.multivariate_normal(mu_ab, Sigma_full, size=200_000)
    mask = np.abs(samples2[:, 1] - x_b_val) < 0.05      # 筛选 x_b≈给定值
    emp_mean = samples2[mask, 0].mean()
    print(f"  条件采样验证：筛选后 x_a 均值 = {emp_mean:.3f}（公式 {mu_cond:.3f}）")
    assert abs(emp_mean - mu_cond) < 0.1, "条件分布公式与采样不符"
    # 边际分布：p(x_a) = N(μ_a, Σ_aa)
    print(f"  边际 p(x_a) 的方差 = Σ_aa = {Sig_aa[0,0]}（公式直接给出，无需计算）")

    # ---- 3.2.6 高斯的贝叶斯定理（共轭先验）----
    print("\n-- 3.2.6 高斯贝叶斯：先验 N(μ|μ0,σ0²) + 数据 -> 后验 N(μ|μN,σN²)")
    mu0, sigma0_2 = 0.0, 1.0       # 先验：均值 μ 的先验分布
    sigma2 = 0.5                   # 数据噪声方差（已知）
    data = rng.normal(1.2, np.sqrt(sigma2), size=5)   # 5 个观测
    xbar = data.mean()
    N_dat = data.size
    # 后验参数（书中公式 3.49-3.51 形式）：
    #   1/σN² = 1/σ0² + N/σ²
    #   μN = σN² ( μ0/σ0² + N x̄/σ² )
    sigmaN2 = 1.0 / (1.0 / sigma0_2 + N_dat / sigma2)
    muN = sigmaN2 * (mu0 / sigma0_2 + N_dat * xbar / sigma2)
    print(f"  先验 μ~N({mu0},{sigma0_2})，观测 x̄={xbar:.3f}（N={N_dat}）")
    print(f"  后验 μ~N({muN:.3f}, {sigmaN2:.3f})：均值从 0 移向数据 1.2，方差变小（信念收紧）")

    # ---- 3.2.7 多元高斯 MLE ----
    print("\n-- 3.2.7 MLE：μ_ML = x̄，Σ_ML = (1/N)Σ(x_n-x̄)(x_n-x̄)ᵀ")
    mu_ml2 = xs.mean(axis=0)
    diff = xs - mu_ml2
    Sigma_ml = (diff.T @ diff) / xs.shape[0]          # 注意除以 N（不是 N-1）
    print(f"  μ_ML = {np.round(mu_ml2, 3)}，Σ_ML = {np.round(Sigma_ml, 3)}")
    print(f"  （对照 np.cov 用 N-1：{np.round(np.cov(xs.T), 3)} —— 与第 2 章的偏差话题呼应）")

    # ---- 3.2.8 顺序估计（Robbins-Monro）----
    print("\n-- 3.2.8 顺序估计：μ_N = μ_{N-1} + (1/N)(x_N - μ_{N-1})")
    seq_data = rng.normal(1.0, 1.0, size=500)
    mu_seq = 0.0
    for i, x in enumerate(seq_data, start=1):
        mu_seq += (1.0 / i) * (x - mu_seq)            # 每来一个数据在线更新均值
    print(f"  顺序估计终值 μ = {mu_seq:.4f}（批处理均值 {seq_data.mean():.4f}，两者等价）")
    assert abs(mu_seq - seq_data.mean()) < 1e-9, "顺序估计与批处理不一致"

    # ---- 3.2.9 高斯混合 ----
    print("\n-- 3.2.9 高斯混合：p(x) = Σ_k π_k N(x|μ_k, Σ_k)")
    pis = np.array([0.4, 0.6])
    mus = np.array([-2.0, 2.0])
    sigmas = np.array([0.5, 0.8])
    xg = np.linspace(-5, 5, 400)
    mix = pis[0] * gaussian_pdf(xg, mus[0], sigmas[0] ** 2) + \
          pis[1] * gaussian_pdf(xg, mus[1], sigmas[1] ** 2)
    fig = Figure("高斯混合 0.4·N(-2,0.25)+0.6·N(2,0.64)", "x", "p(x)")
    fig.line(xg, mix, label="混合密度")
    fig.line(xg, pis[0] * gaussian_pdf(xg, mus[0], sigmas[0] ** 2), label="分量1")
    fig.line(xg, pis[1] * gaussian_pdf(xg, mus[1], sigmas[1] ** 2), label="分量2")
    fig.save(os.path.join(PLOTS_DIR, "fig2_gaussian_mixture.png"))
    # 归一化验证
    mix_int = float(np.sum(mix) * (xg[1] - xg[0]))
    print(f"  混合密度积分 ≈ {mix_int:.4f}（应 ≈1）")
    assert abs(mix_int - 1.0) < 1e-3, "混合密度未归一化"

    # ==================================================================
    # 3.3 周期变量：von Mises（书中 3.3 节）
    # ==================================================================
    section("3.3 周期变量：von Mises 分布（书中 3.3.1 节）")
    # p(θ) = 1/(2π I0(m)) exp(m cos(θ-θ0))，I0 是 0 阶修正贝塞尔函数（numpy 提供）
    def von_mises_pdf(theta, theta0, m):
        """von Mises 密度。theta 用弧度。"""
        return np.exp(m * np.cos(theta - theta0)) / (2 * np.pi * np.i0(m))

    theta_grid = np.linspace(-np.pi, np.pi, 400)
    fig = Figure("von Mises 分布：集中度 m 越大越尖（书中图 3.9 风格）", "θ", "p(θ)")
    for m in (0.0, 1.0, 4.0):
        fig.line(theta_grid, von_mises_pdf(theta_grid, 0.0, m), label=f"m={m}")
    fig.save(os.path.join(PLOTS_DIR, "fig3_von_mises.png"))
    # 归一化验证
    for m in (0.0, 2.0, 5.0):
        val = float(np.sum(von_mises_pdf(theta_grid, 0.0, m)) * (theta_grid[1] - theta_grid[0]))
        print(f"  m={m}：∫p dθ ≈ {val:.4f}（应 ≈1）")
    print("  注意 m=0 时 von Mises 退化为均匀分布（cos 项为 0）")

    # ==================================================================
    # 3.4 指数族（书中 3.4 节）
    # ==================================================================
    section("3.4 指数族：p(x|η) = h(x) g(η) exp(ηᵀ u(x))（书中 3.4 节）")
    print("伯努利分布是指数族：")
    print("  Bern(x|μ) = exp( x ln(μ/(1-μ)) + ln(1-μ) )")
    print("  自然参数 η = ln(μ/(1-μ))，充分统计量 u(x) = x")
    mu_demo = 0.3
    eta = np.log(mu_demo / (1 - mu_demo))
    print(f"  例：μ={mu_demo} -> η={eta:.3f}（logit 变换）")
    # 验证指数族形式还原伯努利
    def bernoulli_expfam(x, eta):
        """用指数族一般形式计算 Bern 概率：exp(ηx) / (1+exp(η))。"""
        return np.exp(eta * x) / (1 + np.exp(eta))
    for x in (0, 1):
        v1 = bernoulli_expfam(x, eta)
        v2 = bernoulli_pmf(x, mu_demo)
        print(f"  x={x}：指数族形式 {v1:.4f} vs 直接公式 {v2:.4f}（一致 {abs(v1-v2)<1e-12}）")
        assert abs(v1 - v2) < 1e-12
    print("  高斯也是指数族：η=[μ/σ², -1/(2σ²)]，u(x)=[x, x²]（充分统计量为一、二阶矩）")

    # ==================================================================
    # 3.5 非参数方法（书中 3.5 节）
    # ==================================================================
    section("3.5 非参数方法：直方图 / 核密度 / k 近邻（书中 3.5 节）")
    # 生成双峰数据做演示
    rng5 = np.random.default_rng(5)
    mix_data = np.concatenate([
        rng5.normal(-1.5, 0.5, 400),
        rng5.normal(1.5, 0.7, 400),
    ])

    # ---- 3.5.1 直方图：箱宽影响 ----
    fig = Figure("直方图：箱宽的影响（书中图 3.16 风格）", "x", "密度")
    for bw in (0.1, 0.5, 2.0):
        hist_y, hist_x = np.histogram(mix_data, bins=np.arange(-4, 4, bw), density=True)
        cc = (hist_x[:-1] + hist_x[1:]) / 2
        fig.line(cc, hist_y, label=f"箱宽={bw}")
    fig.save(os.path.join(PLOTS_DIR, "fig4_histogram.png"))
    print("  箱宽小 -> 噪声大；箱宽大 -> 过度平滑；箱宽影响起点位置（直方图固有缺陷）")

    # ---- 3.5.2 核密度估计：带宽影响 ----
    print("\n-- 核密度：p(x) = (1/N) Σ K_h(x - x_n)，高斯核，带宽 h 影响平滑度")
    def kde(x_grid, data, h):
        """高斯核密度估计：p(x)=1/(N h) Σ φ((x-x_n)/h)。"""
        out = np.zeros_like(x_grid)
        for xn in data:
            out += gaussian_pdf((x_grid - xn) / h, 0.0, 1.0)   # 标准高斯核
        return out / (len(data) * h)
    xk = np.linspace(-4, 4, 400)
    fig = Figure("核密度估计：带宽 h 的影响（书中图 3.17 风格）", "x", "p(x)")
    for h in (0.1, 0.3, 1.0):
        fig.line(xk, kde(xk, mix_data, h), label=f"h={h}")
    fig.save(os.path.join(PLOTS_DIR, "fig5_kde.png"))
    print("  h 小 -> 过拟合噪声；h 大 -> 过平滑；h=0.3 左右能较好呈现双峰")

    # ---- 3.5.3 k 近邻密度 ----
    print("\n-- k 近邻：p(x) = k / (N V)，V 是包含 k 个近邻的球体积")
    # 在 1 维上：V = 2r（半径 r 的"球"）
    def knn_density_1d(x_grid, data, k):
        """一维 k 近邻密度估计。"""
        out = np.zeros_like(x_grid)
        data_sorted = np.sort(data)
        n = len(data_sorted)
        for i, xv in enumerate(x_grid):
            dists = np.abs(data_sorted - xv)
            idx = np.argsort(dists)
            r = dists[idx[k - 1]]                      # 第 k 个近邻的距离
            r = max(r, 1e-6)
            out[i] = k / (n * 2 * r)                   # V = 2r
        return out
    fig = Figure("k 近邻密度估计（书中图 3.18 风格）", "x", "p(x)")
    fig.line(xk, knn_density_1d(xk, mix_data, k=10), label="k=10")
    fig.line(xk, knn_density_1d(xk, mix_data, k=50), label="k=50")
    fig.save(os.path.join(PLOTS_DIR, "fig6_knn.png"))
    print("  k 近邻密度在数据稀疏处平滑下降（V 自适应），与直方图/核密度的平滑方式不同")

    # ==================================================================
    # 4. 本章小结
    # ==================================================================
    section("4. 小结：这一章你亲眼看到了什么")
    print("""
  1. 伯努利/二项/多项分布：离散数据的标准模型，MLE 都很简单（频率）；
  2. 多元高斯：Σ 的特征分解决定椭球几何；条件/边际有闭式公式；
  3. 高斯是自共轭的：先验高斯 + 高斯似然 -> 后验高斯（贝叶斯可解析）；
  4. 单高斯局限明显（单峰），高斯混合可以表达任意复杂分布；
  5. von Mises 处理角度/周期数据；
  6. 指数族统一了伯努利、高斯等：η 与 u(x) 是核心，u 即充分统计量；
  7. 非参数方法（直方图/核密度/kNN）不用假设分布族，但要调超参数
     （箱宽/带宽/k），存在偏差-方差权衡。
""")


if __name__ == "__main__":
    main()
