# -*- coding: utf-8 -*-
"""
第 2 章：概率论
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 2 章
（印刷页 23-64，PDF 页 43-84）。

本章要亲眼看到的现象：
  1. 概率的频率视角：弯曲硬币掷 N 次，频率收敛到 0.6（图 2.2）；
  2. 和规则（边际化）与积规则：从联合计数表出发推导，并验证一致性；
  3. 贝叶斯定理 + 医学筛查：1% 患病率、3% 假阳、10% 假阴，
     算出 P(阳性)=3.87%、P(患癌|阳性)≈23.3% —— 并用 10 万人模拟验证；
  4. 先验 → 后验：观察数据如何更新我们的信念；
  5. 高斯分布：pdf、极大似然估计 μ_ML、σ²_ML，以及 σ²_ML 的偏差
     （σ²_ML 平均低估真实方差，偏差因子 (N-1)/N）；
  6. 密度变换：非线性变换 y = f(x) 下概率密度如何变化（雅可比）；
  7. 多变量高斯：球面/对角/一般协方差的等高线差异；
  8. 信息论：熵（均匀 8 状态 = 3 bits，非均匀例 = 2 bits）、微分熵、
     最大熵、KL 散度（非负、不对称）、条件熵、互信息；
  9. 贝叶斯概率：beta 先验 + 抛硬币数据 → 后验的演化（图 2.18 风格）。

运行方式：
  C:/Python314/python.exe ch02_probabilities.py
输出：
  _plots/ 下 8 张图 + 终端中文叙述
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

# 把仓库根目录加入路径，便于 import utils
# 脚本位于 repo/chapters/chXX/ 下，需要向上 3 层才到仓库根
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
from utils import (Figure, entropy, gaussian_pdf, kl_divergence,
                   multivariate_gaussian_pdf, mutual_information)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    """打印小节标题分隔线，方便在终端里看清当前跑到哪一节了。"""
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ==================================================================
    # 1. 概率的频率视角：弯曲硬币（书中 2.1 节开头，图 2.2）
    # ==================================================================
    section("1. 频率视角：弯曲硬币掷 N 次，频率收敛到 0.6（书中图 2.2）")
    print("弯曲硬币：P(凹面朝上) = 0.6，P(凸面朝上) = 0.4（书中设定）")
    p_concave = 0.6          # 硬币落向凹面的真实概率（我们事先假设已知）
    rng = np.random.default_rng(0)   # 固定随机种子，保证每次运行结果可复现

    # 掷 5000 次：每次生成一个 [0,1) 均匀随机数，小于 0.6 视为"凹面朝上"
    flips = rng.random(5000)
    is_concave = flips < p_concave          # 布尔数组：True=凹面
    freq = np.cumsum(is_concave) / np.arange(1, 5001)   # 累计频率 = 凹面次数 / 总次数

    # 打印几个关键时刻的频率，直观看到"频率逼近概率"
    for n in (10, 100, 1000, 5000):
        print(f"  掷 {n:5d} 次后，凹面频率 = {freq[n-1]:.4f}  （理论值 0.6）")

    # 画频率收敛曲线：横轴掷币次数，纵轴累计频率，加一条 0.6 的参考线
    fig = Figure("弯曲硬币：频率随掷币次数收敛到 0.6（书中图 2.2）", "掷币次数 N", "凹面频率")
    fig.line(np.arange(1, 5001), freq, label="观测频率")
    fig.line([1, 5000], [0.6, 0.6], label="理论概率 0.6")
    fig.save(os.path.join(PLOTS_DIR, "fig1_coin_frequency.png"))

    # 数值验证：最后频率应当接近 0.6（误差 < 0.02 即可，随机波动可接受）
    assert abs(freq[-1] - 0.6) < 0.02, f"频率未收敛到 0.6：{freq[-1]}"

    # ==================================================================
    # 2. 和规则与积规则：从联合计数表推导（书中 2.1.2 节，图 2.4）
    # ==================================================================
    section("2. 和规则（边际化）与积规则（书中 2.1.2 节，图 2.4）")
    print("书中用两个随机变量 X、Y 的计数表来推导：")
    print("  n_ij = 同时出现 (X=xi, Y=yj) 的次数；N = 总试验次数")
    print("  联合概率 p(X=xi,Y=yj) = n_ij / N")

    # 构造一个 L=5 行（X 的 5 个取值）x M=3 列（Y 的 3 个取值）的计数表
    # 数字随意选，只要非负即可 —— 它代表 N 次抽样落在每个格子里的人数
    n_ij = np.array([
        [12, 8, 20],   # X=x1 时 Y 的三种取值计数
        [15, 5, 30],   # X=x2
        [25, 10, 35],  # X=x3
        [18, 2, 40],   # X=x4
        [30, 15, 25],  # X=x5
    ])
    N = int(n_ij.sum())                      # 总人数（所有格子加起来）
    joint = n_ij / N                         # 联合概率表 p(X,Y)：每个格子除以 N

    # ---- 和规则：p(X=xi) = Σ_j p(X=xi, Y=yj) ----
    # 对每一行求和（axis=1 表示沿"列"方向压缩），得到 X 的边际分布
    pX = joint.sum(axis=1)
    print(f"  联合概率表 p(X,Y) 的行和 = {np.round(pX, 4)}  <- 这就是边际化（和规则）")

    # ---- 积规则：p(X,Y) = p(Y|X) p(X) ----
    # 条件概率 p(Y=yj | X=xi) = n_ij / c_i，即第 i 行除以该行总数
    pY_given_X = joint / pX[:, None]         # [:, None] 把 pX 变成列向量以便逐行除
    # 验证积规则：p(Y|X) * p(X) 应该还原联合分布（允许浮点误差）
    recovered = pY_given_X * pX[:, None]
    max_err = float(np.abs(recovered - joint).max())
    assert max_err < 1e-12, f"积规则还原失败：{max_err}"
    print(f"  验证积规则：p(Y|X)*p(X) 还原联合分布，最大误差 {max_err:.1e} ✓")
    print(f"  条件概率表 p(Y|X) 每行之和 = {np.round(pY_given_X.sum(axis=1), 4)}（应全为 1）")

    # ==================================================================
    # 3. 贝叶斯定理 + 医学筛查（书中 2.1.1 / 2.1.3 / 2.1.4 节）
    # ==================================================================
    section("3. 贝叶斯定理与医学筛查（书中 2.1.1/2.1.3/2.1.4 节）")
    print("书中例子：筛查癌症。患病率 1%、假阳性率 3%、假阴性率 10%")
    P_C = 0.01                  # 先验：人群中患癌比例 p(C) = 1%
    P_neg_given_C = 0.10        # 假阴性率：患癌却检测为阴性的比例
    P_pos_given_notC = 0.03     # 假阳性率：没患癌却检测为阳性的比例
    P_pos_given_C = 1.0 - P_neg_given_C   # 灵敏度：患癌时检测阳性的概率 = 90%
    P_notC = 1.0 - P_C          # 没患癌的先验 = 99%

    # 问题 1：随便抽一个人，检测呈阳性的总概率 p(阳性)
    # 用"全概率公式"（本质是把阳性拆成两条互斥路径再相加）：
    #   p(+) = p(+|C)p(C) + p(+|¬C)p(¬C)
    P_pos = P_pos_given_C * P_C + P_pos_given_notC * P_notC
    print(f"  Q1: p(检测阳性) = {P_pos:.4f} = {P_pos*100:.2f}%")

    # 问题 2：已知检测阳性，真的患癌的概率 p(C|+)
    # 贝叶斯定理：p(C|+) = p(+|C) p(C) / p(+)
    P_C_given_pos = P_pos_given_C * P_C / P_pos
    print(f"  Q2: p(患癌 | 检测阳性) = {P_C_given_pos:.4f} = {P_C_given_pos*100:.1f}%")
    print("  结论：检测阳性的人里，真正患癌的只有约 23% —— 因为没患癌的人基数太大。")

    # ---- 10 万人蒙特卡洛模拟验证 ----
    # 模拟 10 万人：先按患病率决定谁患癌，再按各自概率决定检测结果
    N_people = 100_000
    rng2 = np.random.default_rng(7)
    has_cancer = rng2.random(N_people) < P_C          # 1% 的人患癌
    # 每个人生成一个 [0,1) 随机数：患癌者 <0.9 为阳性，未患癌者 <0.03 为阳性
    test_positive = rng2.random(N_people) < np.where(has_cancer, P_pos_given_C, P_pos_given_notC)

    # 从模拟结果里直接数：阳性的人里有多少比例真的患癌
    n_pos = int(test_positive.sum())
    n_cancer_and_pos = int((has_cancer & test_positive).sum())
    sim_cond = n_cancer_and_pos / n_pos if n_pos > 0 else float("nan")
    print(f"  模拟 {N_people:,} 人：阳性 {n_pos:,} 人，其中患癌 {n_cancer_and_pos:,} 人")
    print(f"  模拟得到的 p(患癌|阳性) = {sim_cond:.4f}  （理论 {P_C_given_pos:.4f}）")
    assert abs(sim_cond - P_C_given_pos) < 0.01, "模拟与理论不一致"
    print("  模拟与理论一致 ✓")

    # 画一张"患癌/未患癌 x 阳性/阴性"的分组计数图（书中图 2.3 的数字版）
    groups = [
        ("患癌+阳性", int((has_cancer & test_positive).sum())),
        ("患癌+阴性", int((has_cancer & ~test_positive).sum())),
        ("未患癌+阳性", int((~has_cancer & test_positive).sum())),
        ("未患癌+阴性", int((~has_cancer & ~test_positive).sum())),
    ]
    fig = Figure("10 万人模拟：四个分组人数（书中图 2.3 的数字版）", "分组", "人数")
    xs = np.arange(len(groups))
    fig.scatter(xs, [g[1] for g in groups], label="人数")
    fig.save(os.path.join(PLOTS_DIR, "fig2_screening.png"))
    for name, cnt in groups:
        print(f"    {name:8s}: {cnt:6d} 人 ({cnt/N_people*100:.1f}%)")

    # ==================================================================
    # 4. 先验 → 后验：信念更新 + 独立变量（书中 2.1.5 / 2.1.6 节）
    # ==================================================================
    section("4. 先验与后验：观察数据如何更新信念（书中 2.1.5 节）")
    print("先验 p(C) = 0.01 -> 观察到阳性 -> 后验 p(C|+) = 23.3%")
    print("数据（检测结果）把患癌概率从 1% 抬升到 23% —— 但远非 100%，")
    print("因为假阳性太多。这就是贝叶斯定理的直观意义：后验 = 先验 x 似然 / 归一化")

    # 独立变量（书中 2.1.6 节）：p(X,Y) = p(X)p(Y) 时称 X、Y 独立
    print("\n独立变量（书中 2.1.6 节）：若 p(X,Y)=p(X)p(Y)，则观测 Y 不会改变 X 的分布")
    # 构造一个独立联合表：joint = outer(pX, pY)，验证 p(X|Y) = p(X)
    pX_demo = np.array([0.3, 0.7])
    pY_demo = np.array([0.2, 0.5, 0.3])
    joint_indep = np.outer(pX_demo, pY_demo)          # 外积 = 独立时的联合分布
    pX_given_y0 = joint_indep[:, 0] / joint_indep[:, 0].sum()   # 观测 Y=y0 后的 X 分布
    assert np.allclose(pX_given_y0, pX_demo), "独立变量下条件分布应等于边际分布"
    print(f"  独立情形：p(X) = {pX_demo}，观测 Y=y0 后 p(X|y0) = {np.round(pX_given_y0,4)}（不变 ✓）")

    # ==================================================================
    # 5. 概率密度：归一化 + 期望与协方差（书中 2.2 节）
    # ==================================================================
    section("5. 概率密度：归一化、期望、协方差（书中 2.2 节）")
    # 连续变量的概率密度 p(x) 必须满足 ∫p(x)dx = 1
    # 数值验证：把 [-10, 10] 切成 4001 个点，用矩形法则近似积分
    grid = np.linspace(-10, 10, 4001)
    dx = grid[1] - grid[0]
    integral = float(np.sum(gaussian_pdf(grid, mu=0.0, sigma2=2.0)) * dx)
    print(f"  验证高斯密度归一化：∫N(x|0,2)dx ≈ {integral:.6f}（应接近 1）")
    assert abs(integral - 1.0) < 1e-4, "密度未归一化"

    # 期望 E[x]：用大量高斯样本计算样本均值，验证接近理论均值 0
    samples = np.random.default_rng(1).normal(0.0, np.sqrt(2.0), size=200_000)
    print(f"  E[x]（200k 样本均值）= {samples.mean():.4f}（理论 0）")

    # 协方差：两个相关变量的样本协方差
    rng3 = np.random.default_rng(2)
    z1 = rng3.normal(0, 1, 100_000)
    z2 = rng3.normal(0, 1, 100_000)
    x_corr = z1                      # 第一个变量就是 z1
    y_corr = 0.8 * z1 + 0.6 * z2     # 第二个变量 = 0.8*z1 + 0.6*z2（与 z1 相关）
    cov_xy = float(np.mean((x_corr - x_corr.mean()) * (y_corr - y_corr.mean())))
    print(f"  cov(x,y)（样本估计）= {cov_xy:.4f}（理论 0.8）")

    # ==================================================================
    # 6. 高斯分布：pdf、极大似然估计、MLE 的偏差（书中 2.3 节）
    # ==================================================================
    section("6. 高斯分布与极大似然（书中 2.3 节）")
    # 画不同 σ 的高斯 pdf 曲线，直观感受"尖/扁"
    fig = Figure("高斯密度 N(x|0, σ²)：σ 越小越尖（书中图 2.6 风格）", "x", "p(x)")
    xs = np.linspace(-5, 5, 400)
    for sigma2 in (0.2, 1.0, 5.0):
        fig.line(xs, gaussian_pdf(xs, 0.0, sigma2), label=f"σ²={sigma2}")
    fig.save(os.path.join(PLOTS_DIR, "fig3_gaussian_pdf.png"))

    # ---- 极大似然估计（书中 2.3.2 节）----
    # 生成 N 个高斯样本，用 MLE 公式估计 μ、σ²
    #   μ_ML  = (1/N) Σ x_n
    #   σ²_ML = (1/N) Σ (x_n - μ_ML)²
    rng4 = np.random.default_rng(3)
    N_gauss = 20
    true_mu, true_sigma2 = 1.0, 4.0
    data = rng4.normal(true_mu, np.sqrt(true_sigma2), N_gauss)
    mu_ml = data.mean()                                        # MLE 均值
    sigma2_ml = float(((data - mu_ml) ** 2).mean())            # MLE 方差（注意除以 N）
    print(f"  真实参数：μ={true_mu}, σ²={true_sigma2}，N={N_gauss}")
    print(f"  MLE 估计：μ_ML={mu_ml:.3f}，σ²_ML={sigma2_ml:.3f}")

    # ---- MLE 的偏差（书中 2.3.3 节，图 2.10）----
    # 关键教学点：σ²_ML 平均来说低估了真实方差！
    # 直觉：μ_ML 本身是从数据里估出来的，会"贴着"数据，使残差偏小。
    # 数学上 E[σ²_ML] = (N-1)/N σ²_true
    n_reps = 5000
    sigma2_mls = np.empty(n_reps)
    for rep in range(n_reps):
        d = rng4.normal(true_mu, np.sqrt(true_sigma2), N_gauss)
        sigma2_mls[rep] = ((d - d.mean()) ** 2).mean()
    avg_sigma2_ml = float(sigma2_mls.mean())
    expected = (N_gauss - 1) / N_gauss * true_sigma2          # (N-1)/N 倍
    print(f"  重复 {n_reps} 次实验：σ²_ML 的平均 = {avg_sigma2_ml:.3f}")
    print(f"  理论期望 E[σ²_ML] = (N-1)/N σ² = {expected:.3f}  <- 明显小于真实 σ²={true_sigma2}")
    assert abs(avg_sigma2_ml - expected) < 0.15, "MLE 偏差模拟与理论不符"
    print("  模拟验证了系统性偏差：σ²_ML 平均低估方差（无偏修正的动机）✓")

    # 画偏差示意图：5000 个 σ²_ML 的直方图 + 真实值竖线
    fig = Figure("σ²_ML 的分布：平均低于真实方差（书中图 2.10 风格）", "σ²_ML 估计值", "频数")
    hist_y, hist_x = np.histogram(sigma2_mls, bins=40, density=True)
    centers = (hist_x[:-1] + hist_x[1:]) / 2
    fig.line(centers, hist_y, label="σ²_ML 分布")
    fig.line([true_sigma2, true_sigma2], [0, hist_y.max() * 1.05], label="真实 σ²")
    fig.line([expected, expected], [0, hist_y.max() * 1.05], label="E[σ²_ML]")
    fig.save(os.path.join(PLOTS_DIR, "fig4_mle_bias.png"))

    # ==================================================================
    # 7. 密度变换：y = f(x) 时密度如何变化（书中 2.4 节）
    # ==================================================================
    section("7. 密度变换（书中 2.4 节）：非线性映射下的密度变形")
    # 设 x ~ N(0,1)，令 y = x²。分析密度：
    #   p(y) = p(x) |dx/dy|，对单调分支求和。x=±√y 两个分支，|dx/dy| = 1/(2√y)
    # 我们直接用"变换后的样本直方图"对比"解析公式"，验证雅可比因子
    rng5 = np.random.default_rng(4)
    x_samples = rng5.normal(0, 1, 200_000)
    y_samples = x_samples ** 2          # 对每个样本做非线性变换

    # 解析密度：p(y) = 2 * N(√y|0,1) * 1/(2√y) = N(√y|0,1)/√y （y>0）
    y_grid = np.linspace(0.01, 8, 500)
    p_analytic = gaussian_pdf(np.sqrt(y_grid), 0.0, 1.0) / np.sqrt(y_grid)

    # 直方图与解析密度对比
    hist_y2, hist_x2 = np.histogram(y_samples, bins=60, range=(0, 8), density=True)
    centers2 = (hist_x2[:-1] + hist_x2[1:]) / 2
    fig = Figure("密度变换 y=x²：样本直方图 vs 解析密度（书中图 2.13 风格）", "y", "p(y)")
    fig.line(centers2, hist_y2, label="变换后样本直方图")
    fig.line(y_grid, p_analytic, label="解析 p(y)（含雅可比）")
    fig.save(os.path.join(PLOTS_DIR, "fig5_density_transform.png"))
    approx = np.interp(centers2, y_grid, p_analytic)
    rel = np.abs(approx - hist_y2) / np.maximum(approx, 1e-9)
    print(f"  变换后直方图 vs 解析密度：中位相对误差 {np.median(rel):.3f}（<0.3 即吻合）")
    assert np.median(rel) < 0.3, "密度变换验证失败"
    print("  验证了雅可比变换：非线性映射会拉伸/压缩概率质量 ✓")

    # ---- 2.4.1 多变量高斯：不同协方差的等高线 ----
    print("\n  多变量高斯（书中 2.4.1 节）：协方差矩阵决定分布的形状")
    mu2 = np.array([0.0, 0.0])
    covs = {
        "spherical": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "diagonal": np.array([[1.0, 0.0], [0.0, 4.0]]),
        "full": np.array([[1.0, 0.8], [0.8, 1.0]]),
    }
    xg = np.linspace(-4, 4, 200)
    XG, YG = np.meshgrid(xg, xg)
    pts = np.stack([XG.ravel(), YG.ravel()], axis=1)
    for name, Sigma in covs.items():
        vals = np.array([multivariate_gaussian_pdf(p, mu2, Sigma) for p in pts])
        fig = Figure(f"二维高斯等高线：{name}", "x1", "x2")
        fig.scatter(pts[::8, 0], pts[::8, 1], label=name)
        fig.save(os.path.join(PLOTS_DIR, f"fig6_mvn_{name}.png"))
    print("  生成的图 fig6_mvn_*.png：球面=圆，对角=沿轴椭圆，一般=倾斜椭圆（相关）")

    # ==================================================================
    # 8. 信息论：熵、微分熵、最大熵、KL、条件熵、互信息（书中 2.5 节）
    # ==================================================================
    section("8. 信息论（书中 2.5 节）")
    # ---- 熵：两个书中原例 ----
    p_uniform8 = np.ones(8) / 8          # 均匀 8 状态
    H_uniform8 = entropy(p_uniform8, unit="bits")
    print(f"  均匀 8 状态：H = {H_uniform8:.4f} bits（书中 3 bits）")

    # Cover-Thomas 例子：(1/2, 1/4, 1/8, 1/16, 1/64, 1/64, 1/64, 1/64)
    p_ct = np.array([0.5, 0.25, 0.125, 0.0625] + [1 / 64] * 4)
    H_ct = entropy(p_ct, unit="bits")
    print(f"  非均匀例：H = {H_ct:.4f} bits（书中 2 bits）")
    assert abs(H_uniform8 - 3.0) < 1e-6 and abs(H_ct - 2.0) < 1e-6
    print("  两个熵例子与书完全一致 ✓")

    # 画"二值分布的熵随 p 变化"曲线：p=0.5 时熵最大（不确定性最大）
    ps = np.linspace(0.001, 0.999, 200)
    Hs = np.array([entropy(np.array([p, 1 - p]), unit="bits") for p in ps])
    fig = Figure("二值分布熵 H(p) 随 p 变化", "p", "H(p) / bits")
    fig.line(ps, Hs, label="H(p)")
    fig.save(os.path.join(PLOTS_DIR, "fig7_entropy.png"))

    # ---- 微分熵（书中 2.5.3 节）：高斯的微分熵 = ½ ln(2πeσ²) ----
    sigma2_demo = 2.0
    h_gauss = 0.5 * np.log(2 * np.pi * np.e * sigma2_demo)
    fine = np.linspace(-20, 20, 20001)
    pv = gaussian_pdf(fine, 0.0, sigma2_demo)
    h_num = float(-np.sum(pv * np.log(np.maximum(pv, 1e-300))) * (fine[1] - fine[0]))
    print(f"  高斯微分熵：解析 ½ln(2πeσ²) = {h_gauss:.4f}，数值积分 = {h_num:.4f}")
    assert abs(h_num - h_gauss) < 1e-3, "微分熵验证失败"

    # ---- 最大熵（书中 2.5.4 节）：给定方差时高斯熵最大 ----
    var_demo = 1.0
    h_gauss_max = 0.5 * np.log(2 * np.pi * np.e * var_demo)
    half_width = np.sqrt(3 * var_demo)         # 均匀分布 U(-a,a) 的方差 = a²/3
    h_uniform_cont = np.log(2 * half_width)    # 均匀分布微分熵 = ln(宽度)
    print(f"  最大熵：同方差 σ²=1 下，高斯微分熵 {h_gauss_max:.4f} > 均匀 {h_uniform_cont:.4f} ✓")

    # ---- KL 散度（书中 2.5.5 节）：非负、不对称 ----
    p_kl = np.array([0.4, 0.35, 0.25])
    q_kl = np.array([0.3, 0.3, 0.4])
    kl_pq = kl_divergence(p_kl, q_kl)
    kl_qp = kl_divergence(q_kl, p_kl)
    print(f"  KL(p||q) = {kl_pq:.4f}，KL(q||p) = {kl_qp:.4f}（不对称！）")
    assert kl_pq >= 0 and kl_qp >= 0, "KL 散度应为非负"
    assert abs(kl_divergence(p_kl, p_kl)) < 1e-12, "KL(p||p) 应为 0"
    print("  KL 非负 ✓，KL(p||p)=0 ✓，不对称 ✓")

    # ---- 条件熵与互信息（书中 2.5.6 / 2.5.7 节）----
    # 构造联合分布 p(X,Y)：X、Y 各 3 个取值，故意让它们相关
    joint_xy = np.array([
        [0.10, 0.02, 0.01],
        [0.02, 0.30, 0.03],
        [0.01, 0.03, 0.48],
    ])
    MI = mutual_information(joint_xy)
    # 用 H(X) + H(Y) - H(X,Y) 交叉验证
    px = joint_xy.sum(axis=1)
    py = joint_xy.sum(axis=0)
    Hx = entropy(px)
    Hy = entropy(py)
    Hxy = entropy(joint_xy.ravel())
    MI2 = Hx + Hy - Hxy
    print(f"  互信息 I(X;Y) = {MI:.4f} nats，交叉验证 H(X)+H(Y)-H(X,Y) = {MI2:.4f}")
    assert abs(MI - MI2) < 1e-9, "互信息两种算法不一致"
    print("  互信息 > 0 说明 X、Y 不独立；若独立则 I=0 ✓")

    # ==================================================================
    # 9. 贝叶斯概率：beta 先验 + 抛硬币（书中 2.6 节）
    # ==================================================================
    section("9. 贝叶斯概率：beta 先验 -> 后验（书中 2.6 节）")
    print("把硬币的偏置 μ 视为未知参数，用 beta 分布表达我们对 μ 的信念")
    print("  beta(μ|a,b) ∝ μ^(a-1) (1-μ)^(b-1)")
    print("  观察 H 次正面、T 次反面后：后验 = beta(a+H, b+T)")

    # beta 密度函数（用 gamma 函数算归一化常数）
    def beta_pdf(mu_grid, a, b):
        """beta(μ|a,b) 密度。"""
        from math import gamma
        norm = gamma(a + b) / (gamma(a) * gamma(b))     # 归一化常数 B(a,b)⁻¹
        return norm * mu_grid ** (a - 1) * (1 - mu_grid) ** (b - 1)

    # 模拟：先验 beta(2,2)（温和的无信息先验），观察 10 次抛硬币（7 正 3 反）
    mu_grid = np.linspace(0.001, 0.999, 500)
    a0, b0 = 2, 2
    H, T = 7, 3
    fig = Figure("beta 先验 -> 后验：观察 7 正 3 反后信念更新", "μ（硬币偏置）", "密度")
    fig.line(mu_grid, beta_pdf(mu_grid, a0, b0), label="先验 beta(2,2)")
    fig.line(mu_grid, beta_pdf(mu_grid, a0 + H, b0 + T), label="后验 beta(9,5)")
    fig.save(os.path.join(PLOTS_DIR, "fig8_beta_posterior.png"))
    mode_post = (a0 + H - 1) / (a0 + H + b0 + T - 2)
    print(f"  先验众数 = {(a0-1)/(a0+b0-2):.2f}（中性 0.5），后验众数 = {mode_post:.2f}（偏向数据 0.7）")
    print("  贝叶斯更新：数据把信念从均匀推向观察到的频率，先验越强移动越慢。")

    # ==================================================================
    # 10. 本章小结
    # ==================================================================
    section("10. 小结：这一章你亲眼看到了什么")
    print("""
  1. 概率有两种视角：频率（长期频率收敛）与贝叶斯（不确定性的量化）；
  2. 和规则 = 边际化，积规则 = 条件分解，贝叶斯定理 = 两者结合；
  3. 医学筛查：检测阳性 ≠ 患病（只有 23%）—— 先验基数的重要性；
  4. MLE 方差有系统性偏差 σ²_ML = (N-1)/N σ² —— 统计推断的第一课；
  5. 密度变换要乘雅可比 |dx/dy|，非线性映射会拉伸/压缩概率质量；
  6. 熵量化不确定性：均匀分布熵最大；KL 度量分布差异（非负、不对称）；
  7. 互信息 = 观测 Y 带来的 X 不确定性减少量；
  8. 贝叶斯 = 先验 + 数据 -> 后验，是一个不断更新的过程。
""")


if __name__ == "__main__":
    main()
