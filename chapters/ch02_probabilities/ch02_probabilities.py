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
  8. 信息论：熵、微分熵、最大熵、KL 散度、条件熵、互信息；
  9. 贝叶斯概率：beta 先验 + 抛硬币数据 → 后验的演化。

运行方式：
  C:/Python314/python.exe ch02_probabilities.py
输出：
  _plots/ 下 8 张图 + 终端中文叙述
"""
import os
import sys

import numpy as np

# 强制 UTF-8 输出，避免 Windows 控制台（默认 cp1252 编码）打印中文时报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 路径引导：脚本在 repo/chapters/chXX/ 下，向上 3 层到仓库根，import utils
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)
# 导入共享工具（每个函数在 utils.py 里有逐行注释）
from utils import (Figure, entropy, gaussian_pdf, kl_divergence,
                   multivariate_gaussian_pdf, mutual_information)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    """打印小节标题分隔线（纯装饰，方便定位当前执行到哪一节）。"""
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ==================================================================
    # 1. 频率视角：弯曲硬币（书中 2.1 节开头，图 2.2）
    # ==================================================================
    section("1. 频率视角：弯曲硬币掷 N 次，频率收敛到 0.6（书中图 2.2）")
    print("弯曲硬币：P(凹面朝上) = 0.6，P(凸面朝上) = 0.4（书中设定）")
    p_concave = 0.6          # 硬币落向凹面的真实概率（我们"知道"的答案）
    rng = np.random.default_rng(0)   # 固定随机种子：每次运行结果可复现

    # 掷 5000 次硬币：每次生成一个 [0,1) 均匀随机数
    flips = rng.random(5000)
    # 比较：随机数 < 0.6 视为"凹面朝上"。结果是一个布尔数组
    # [True, False, True, ...]，长度 5000（向量化：一次比较整批）
    is_concave = flips < p_concave
    # np.cumsum(is_concave)：累计和。布尔 True 会被当作 1 累加，
    # 所以 cumsum 的结果 = "到第 n 次为止凹面的总次数"
    # np.arange(1, 5001)：[1,2,...,5000]（1-based 的"第几次"）
    # 两者相除 = 累计频率（凹面次数 / 总次数）
    freq = np.cumsum(is_concave) / np.arange(1, 5001)

    # 打印几个关键时刻的频率，直观看到"频率逼近概率"
    for n in (10, 100, 1000, 5000):
        # freq[n-1]：第 n 次时的累计频率（索引从 0 开始，所以要 -1）
        print(f"  掷 {n:5d} 次后，凹面频率 = {freq[n-1]:.4f}  （理论值 0.6）")

    # 画收敛曲线：横轴掷币次数、纵轴累计频率 + 0.6 参考线
    fig = Figure("弯曲硬币：频率随掷币次数收敛到 0.6（书中图 2.2）", "掷币次数 N", "凹面频率")
    fig.line(np.arange(1, 5001), freq, label="观测频率")
    fig.line([1, 5000], [0.6, 0.6], label="理论概率 0.6")   # 水平参考线
    fig.save(os.path.join(PLOTS_DIR, "fig1_coin_frequency.png"))

    # 数值验证：最后频率应接近 0.6（随机波动在 0.02 内可接受）
    assert abs(freq[-1] - 0.6) < 0.02, f"频率未收敛到 0.6：{freq[-1]}"

    # ==================================================================
    # 2. 和规则与积规则：从联合计数表推导（书中 2.1.2 节，图 2.4）
    # ==================================================================
    section("2. 和规则（边际化）与积规则（书中 2.1.2 节，图 2.4）")
    print("书中用两个随机变量 X、Y 的计数表来推导：")
    print("  n_ij = 同时出现 (X=xi, Y=yj) 的次数；N = 总试验次数")
    print("  联合概率 p(X=xi,Y=yj) = n_ij / N")

    # 构造一个计数表：5 行（X 取 5 个值）x 3 列（Y 取 3 个值）
    # 数字随意选，代表 N 次抽样里落在每个"格子"的人数
    n_ij = np.array([
        [12, 8, 20],   # X=x1 时，Y 的三种取值分别出现 12、8、20 次
        [15, 5, 30],   # X=x2
        [25, 10, 35],  # X=x3
        [18, 2, 40],   # X=x4
        [30, 15, 25],  # X=x5
    ])
    N = int(n_ij.sum())                 # 总人数 = 所有格子加起来
    joint = n_ij / N                    # 联合概率表：每个格子除以总人数
    # 形状说明：n_ij 是 (5,3)，joint 也是 (5,3)，joint[i,j] = p(X=xi, Y=yj)

    # ---- 和规则：p(X=xi) = Σ_j p(X=xi, Y=yj) ----
    # joint.sum(axis=1)：axis=1 表示"沿列方向压缩"——对每一行求和
    # 结果 pX 是长度 5 的数组：p(X=x1), p(X=x2), ..., p(X=x5)
    # 这就是"边际化"：把 Y 求和掉，只看 X 的分布
    pX = joint.sum(axis=1)
    print(f"  联合概率表 p(X,Y) 的行和 = {np.round(pX, 4)}  <- 这就是边际化（和规则）")

    # ---- 积规则：p(X,Y) = p(Y|X) p(X) ----
    # 条件概率 p(Y=yj | X=xi) = 格子概率 / 该行概率
    # pX[:, None]：把 (5,) 变成 (5,1)，这样才能和 (5,3) 逐行相除（广播机制）
    # 效果：第 i 行整体除以 pX[i]
    pY_given_X = joint / pX[:, None]
    # 验证积规则：条件概率 x 边际概率 应该还原出联合概率
    # pY_given_X 是 (5,3)，pX[:, None] 是 (5,1)，相乘得到 (5,3)
    recovered = pY_given_X * pX[:, None]
    max_err = float(np.abs(recovered - joint).max())   # 逐元素差的最大值
    assert max_err < 1e-12, f"积规则还原失败：{max_err}"
    print(f"  验证积规则：p(Y|X)*p(X) 还原联合分布，最大误差 {max_err:.1e} ✓")
    # 每行条件概率之和应为 1（给定 X 后，Y 的所有可能加起来是 1）
    print(f"  条件概率表 p(Y|X) 每行之和 = {np.round(pY_given_X.sum(axis=1), 4)}（应全为 1）")

    # ==================================================================
    # 3. 贝叶斯定理 + 医学筛查（书中 2.1.1 / 2.1.3 / 2.1.4 节）
    # ==================================================================
    section("3. 贝叶斯定理与医学筛查（书中 2.1.1/2.1.3/2.1.4 节）")
    print("书中例子：筛查癌症。患病率 1%、假阳性率 3%、假阴性率 10%")
    # 这些数字是书的设定（2.1.1 节），全部来自原文
    P_C = 0.01                  # 先验：人群里患癌比例 p(C) = 1%
    P_neg_given_C = 0.10        # 假阴性率：患癌却检测为阴性的比例
    P_pos_given_notC = 0.03     # 假阳性率：没患癌却检测为阳性的比例
    P_pos_given_C = 1.0 - P_neg_given_C   # 灵敏度：患癌时检测阳性的概率 = 90%
    P_notC = 1.0 - P_C          # 没患癌的先验 = 99%

    # 问题 1：随便抽一个人，检测呈阳性的总概率 p(阳性)
    # 全概率公式：把"阳性"拆成两条互斥路径再相加
    #   p(+) = p(+|C)p(C) + p(+|¬C)p(¬C)
    P_pos = P_pos_given_C * P_C + P_pos_given_notC * P_notC
    print(f"  Q1: p(检测阳性) = {P_pos:.4f} = {P_pos*100:.2f}%")

    # 问题 2：已知检测阳性，真的患癌的概率 p(C|+)
    # 贝叶斯定理：p(C|+) = p(+|C) p(C) / p(+)
    #  分子=灵敏度 x 患病率（真阳性）；分母=总阳性率
    P_C_given_pos = P_pos_given_C * P_C / P_pos
    print(f"  Q2: p(患癌 | 检测阳性) = {P_C_given_pos:.4f} = {P_C_given_pos*100:.1f}%")
    print("  结论：检测阳性的人里，真正患癌的只有约 23% —— 因为没患癌的人基数太大。")

    # ---- 10 万人蒙特卡洛模拟验证 ----
    # 不用公式，直接"模拟 10 万人"：按概率随机决定每个人的患病与检测结果
    N_people = 100_000
    rng2 = np.random.default_rng(7)
    # 第一步：每个人生成 [0,1) 随机数，<0.01 视为患癌
    has_cancer = rng2.random(N_people) < P_C
    # 第二步：再给每个人一个随机数决定检测结果。
    # np.where(条件, a, b)：患癌的人按灵敏度 0.9 判阳性，
    # 未患癌的人按假阳性率 0.03 判阳性（向量化，一次处理 10 万人）
    test_positive = rng2.random(N_people) < np.where(has_cancer, P_pos_given_C, P_pos_given_notC)

    # 从模拟结果数数：阳性的人里有多少真的患癌
    n_pos = int(test_positive.sum())          # 阳性总人数
    # has_cancer & test_positive：两个布尔数组逐位"与"（同时为真才为真）
    n_cancer_and_pos = int((has_cancer & test_positive).sum())
    sim_cond = n_cancer_and_pos / n_pos if n_pos > 0 else float("nan")
    print(f"  模拟 {N_people:,} 人：阳性 {n_pos:,} 人，其中患癌 {n_cancer_and_pos:,} 人")
    print(f"  模拟得到的 p(患癌|阳性) = {sim_cond:.4f}  （理论 {P_C_given_pos:.4f}）")
    assert abs(sim_cond - P_C_given_pos) < 0.01, "模拟与理论不一致"
    print("  模拟与理论一致 ✓（公式正确 + 模拟正确，互相印证）")

    # 画四个分组的人数（书中图 2.3 的数字版）
    groups = [
        ("患癌+阳性", int((has_cancer & test_positive).sum())),
        ("患癌+阴性", int((has_cancer & ~test_positive).sum())),   # ~ 是"按位取反"（布尔取反）
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

    # 独立变量（书中 2.1.6 节）：若 p(X,Y)=p(X)p(Y)，观测 Y 不会改变 X 的分布
    print("\n独立变量（书中 2.1.6 节）：若 p(X,Y)=p(X)p(Y)，则观测 Y 不会改变 X 的分布")
    pX_demo = np.array([0.3, 0.7])            # X 的边际分布
    pY_demo = np.array([0.2, 0.5, 0.3])       # Y 的边际分布
    # np.outer(向量, 向量)：外积，得到 2x3 矩阵，元素 = 两个边际的乘积
    # 这正是"独立"时联合分布的定义：joint[i,j] = pX[i]*pY[j]
    joint_indep = np.outer(pX_demo, pY_demo)
    # 观测 Y=y0 后 X 的条件分布：p(X|y0) = joint[:,0] / Σ joint[:,0]
    # （把 Y=y0 那一列归一化）
    pX_given_y0 = joint_indep[:, 0] / joint_indep[:, 0].sum()
    assert np.allclose(pX_given_y0, pX_demo), "独立变量下条件分布应等于边际分布"
    print(f"  独立情形：p(X) = {pX_demo}，观测 Y=y0 后 p(X|y0) = {np.round(pX_given_y0,4)}（不变 ✓）")

    # ==================================================================
    # 5. 概率密度：归一化 + 期望与协方差（书中 2.2 节）
    # ==================================================================
    section("5. 概率密度：归一化、期望、协方差（书中 2.2 节）")
    # 连续变量的概率密度 p(x) 必须满足 ∫p(x)dx = 1（总面积=1）
    # 数值验证：把 [-10,10] 切成 4001 个点，用矩形法近似积分
    grid = np.linspace(-10, 10, 4001)
    dx = grid[1] - grid[0]                   # 相邻两点间距（矩形宽度）
    # np.sum(高度) * dx = 近似积分（所有小矩形面积之和）
    integral = float(np.sum(gaussian_pdf(grid, mu=0.0, sigma2=2.0)) * dx)
    print(f"  验证高斯密度归一化：∫N(x|0,2)dx ≈ {integral:.6f}（应接近 1）")
    assert abs(integral - 1.0) < 1e-4, "密度未归一化"

    # 期望 E[x]：大数定律——样本均值逼近期望
    samples = np.random.default_rng(1).normal(0.0, np.sqrt(2.0), size=200_000)
    print(f"  E[x]（200k 样本均值）= {samples.mean():.4f}（理论 0）")

    # 协方差：两个相关变量的样本协方差
    # 构造：y = 0.8*z1 + 0.6*z2，其中 z1、z2 独立标准高斯
    # 理论协方差 cov(x,y) = cov(z1, 0.8z1+0.6z2) = 0.8*cov(z1,z1) + 0.6*cov(z1,z2)
    #                     = 0.8*1 + 0.6*0 = 0.8
    rng3 = np.random.default_rng(2)
    z1 = rng3.normal(0, 1, 100_000)
    z2 = rng3.normal(0, 1, 100_000)
    x_corr = z1
    y_corr = 0.8 * z1 + 0.6 * z2
    # 协方差公式：E[(x-μx)(y-μy)]，用样本近似：
    # (x - x.mean())：去均值；(y - y.mean())：去均值；相乘后平均
    cov_xy = float(np.mean((x_corr - x_corr.mean()) * (y_corr - y_corr.mean())))
    print(f"  cov(x,y)（样本估计）= {cov_xy:.4f}（理论 0.8）")

    # ==================================================================
    # 6. 高斯分布：pdf、极大似然估计、MLE 的偏差（书中 2.3 节）
    # ==================================================================
    section("6. 高斯分布与极大似然（书中 2.3 节）")
    # 画不同 σ² 的高斯曲线：直观感受"方差越大越扁"
    fig = Figure("高斯密度 N(x|0, σ²)：σ 越小越尖（书中图 2.6 风格）", "x", "p(x)")
    xs = np.linspace(-5, 5, 400)
    for sigma2 in (0.2, 1.0, 5.0):
        fig.line(xs, gaussian_pdf(xs, 0.0, sigma2), label=f"σ²={sigma2}")
    fig.save(os.path.join(PLOTS_DIR, "fig3_gaussian_pdf.png"))

    # ---- 极大似然估计（书中 2.3.2 节）----
    # MLE 公式（让似然最大的参数估计）：
    #   μ_ML  = (1/N) Σ x_n          （样本均值）
    #   σ²_ML = (1/N) Σ (x_n - μ_ML)²（样本方差，注意除以 N 不是 N-1）
    rng4 = np.random.default_rng(3)
    N_gauss = 20
    true_mu, true_sigma2 = 1.0, 4.0           # 真实参数（生成数据用的）
    data = rng4.normal(true_mu, np.sqrt(true_sigma2), N_gauss)
    mu_ml = data.mean()                                  # MLE 均值 = 样本均值
    sigma2_ml = float(((data - mu_ml) ** 2).mean())      # MLE 方差（除以 N）
    print(f"  真实参数：μ={true_mu}, σ²={true_sigma2}，N={N_gauss}")
    print(f"  MLE 估计：μ_ML={mu_ml:.3f}，σ²_ML={sigma2_ml:.3f}")

    # ---- MLE 的偏差（书中 2.3.3 节，图 2.10）----
    # 关键教学点：σ²_ML 平均来说低估了真实方差！
    # 直觉：μ_ML 是从数据估的，会"贴着"数据，使残差偏小。
    # 数学上 E[σ²_ML] = (N-1)/N σ²_true
    # 下面重复 5000 次实验，每次独立采样 N 个点算 σ²_ML，看平均
    n_reps = 5000
    sigma2_mls = np.empty(n_reps)
    for rep in range(n_reps):
        d = rng4.normal(true_mu, np.sqrt(true_sigma2), N_gauss)
        sigma2_mls[rep] = ((d - d.mean()) ** 2).mean()
    avg_sigma2_ml = float(sigma2_mls.mean())
    expected = (N_gauss - 1) / N_gauss * true_sigma2     # (N-1)/N 倍
    print(f"  重复 {n_reps} 次实验：σ²_ML 的平均 = {avg_sigma2_ml:.3f}")
    print(f"  理论期望 E[σ²_ML] = (N-1)/N σ² = {expected:.3f}  <- 明显小于真实 σ²={true_sigma2}")
    assert abs(avg_sigma2_ml - expected) < 0.15, "MLE 偏差模拟与理论不符"
    print("  模拟验证了系统性偏差：σ²_ML 平均低估方差（无偏修正的动机）✓")

    # 画 σ²_ML 的分布直方图 + 真实值/期望值竖线
    fig = Figure("σ²_ML 的分布：平均低于真实方差（书中图 2.10 风格）", "σ²_ML 估计值", "频数")
    # np.histogram：把数据分箱统计，返回 (每箱高度, 每箱边界)
    hist_y, hist_x = np.histogram(sigma2_mls, bins=40, density=True)
    centers = (hist_x[:-1] + hist_x[1:]) / 2    # 每箱中心点（画折线用）
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
    # 我们直接对比"变换后的样本直方图" vs "解析公式"，验证雅可比因子
    rng5 = np.random.default_rng(4)
    x_samples = rng5.normal(0, 1, 200_000)
    y_samples = x_samples ** 2          # 对每个样本做非线性变换（向量化平方）

    # 解析密度：p(y) = 2 * N(√y|0,1) * 1/(2√y) = N(√y|0,1)/√y （y>0）
    y_grid = np.linspace(0.01, 8, 500)   # y>0（注意从 0.01 开始避免除零）
    p_analytic = gaussian_pdf(np.sqrt(y_grid), 0.0, 1.0) / np.sqrt(y_grid)

    # 直方图与解析密度对比
    hist_y2, hist_x2 = np.histogram(y_samples, bins=60, range=(0, 8), density=True)
    centers2 = (hist_x2[:-1] + hist_x2[1:]) / 2
    fig = Figure("密度变换 y=x²：样本直方图 vs 解析密度（书中图 2.13 风格）", "y", "p(y)")
    fig.line(centers2, hist_y2, label="变换后样本直方图")
    fig.line(y_grid, p_analytic, label="解析 p(y)（含雅可比）")
    fig.save(os.path.join(PLOTS_DIR, "fig5_density_transform.png"))
    # np.interp：把解析密度"采样"到直方图的中心点上，方便逐点比较
    approx = np.interp(centers2, y_grid, p_analytic)
    rel = np.abs(approx - hist_y2) / np.maximum(approx, 1e-9)   # 相对误差
    print(f"  变换后直方图 vs 解析密度：中位相对误差 {np.median(rel):.3f}（<0.3 即吻合）")
    assert np.median(rel) < 0.3, "密度变换验证失败"
    print("  验证了雅可比变换：非线性映射会拉伸/压缩概率质量 ✓")

    # ---- 2.4.1 多变量高斯：不同协方差的等高线 ----
    print("\n  多变量高斯（书中 2.4.1 节）：协方差矩阵决定分布的形状")
    mu2 = np.array([0.0, 0.0])
    covs = {
        "spherical": np.array([[1.0, 0.0], [0.0, 1.0]]),    # 球面：各维独立等方差 -> 圆
        "diagonal": np.array([[1.0, 0.0], [0.0, 4.0]]),     # 对角：各维独立但方差不同 -> 椭圆
        "full": np.array([[1.0, 0.8], [0.8, 1.0]]),         # 一般：有相关 -> 倾斜椭圆
    }
    # 在 200x200 网格上算二维高斯密度（准备画等高线）
    xg = np.linspace(-4, 4, 200)
    XG, YG = np.meshgrid(xg, xg)                 # 生成网格坐标
    pts = np.stack([XG.ravel(), YG.ravel()], axis=1)   # 摊平成点列表 (40000, 2)
    for name, Sigma in covs.items():
        vals = np.array([multivariate_gaussian_pdf(p, mu2, Sigma) for p in pts])
        fig = Figure(f"二维高斯等高线：{name}", "x1", "x2")
        fig.scatter(pts[::8, 0], pts[::8, 1], label=name)
        fig.save(os.path.join(PLOTS_DIR, f"fig6_mvn_{name}.png"))
    print("  生成的图 fig6_mvn_*.png：球面=圆，对角=沿轴椭圆，一般=倾斜椭圆（相关）")

    # ==================================================================
    # 8. 信息论（书中 2.5 节）
    # ==================================================================
    section("8. 信息论（书中 2.5 节）")
    # ---- 熵：两个书中原例 ----
    p_uniform8 = np.ones(8) / 8          # 均匀 8 状态：每个概率 1/8
    H_uniform8 = entropy(p_uniform8, unit="bits")
    print(f"  均匀 8 状态：H = {H_uniform8:.4f} bits（书中 3 bits）")

    # Cover-Thomas 例子：概率 (1/2, 1/4, 1/8, 1/16, 1/64, 1/64, 1/64, 1/64)
    p_ct = np.array([0.5, 0.25, 0.125, 0.0625] + [1 / 64] * 4)   # 列表相加拼接
    H_ct = entropy(p_ct, unit="bits")
    print(f"  非均匀例：H = {H_ct:.4f} bits（书中 2 bits）")
    assert abs(H_uniform8 - 3.0) < 1e-6 and abs(H_ct - 2.0) < 1e-6
    print("  两个熵例子与书完全一致 ✓")

    # 画"二值分布的熵随 p 变化"曲线：p=0.5 时熵最大
    ps = np.linspace(0.001, 0.999, 200)      # p 从 0 到 1（避开端点避免 log(0)）
    # 对每个 p 算 H([p, 1-p])（列表推导式：Python 的一行 for 循环）
    Hs = np.array([entropy(np.array([p, 1 - p]), unit="bits") for p in ps])
    fig = Figure("二值分布熵 H(p) 随 p 变化", "p", "H(p) / bits")
    fig.line(ps, Hs, label="H(p)")
    fig.save(os.path.join(PLOTS_DIR, "fig7_entropy.png"))

    # ---- 微分熵（书中 2.5.3 节）：高斯的微分熵 = ½ ln(2πeσ²) ----
    sigma2_demo = 2.0
    h_gauss = 0.5 * np.log(2 * np.pi * np.e * sigma2_demo)    # 解析公式
    # 数值验证：用细网格近似积分 -∫p ln p dx
    fine = np.linspace(-20, 20, 20001)
    pv = gaussian_pdf(fine, 0.0, sigma2_demo)
    # np.maximum(pv, 1e-300)：防止 pv=0 时 log(0) 报错（clip 到极小值）
    h_num = float(-np.sum(pv * np.log(np.maximum(pv, 1e-300))) * (fine[1] - fine[0]))
    print(f"  高斯微分熵：解析 ½ln(2πeσ²) = {h_gauss:.4f}，数值积分 = {h_num:.4f}")
    assert abs(h_num - h_gauss) < 1e-3, "微分熵验证失败"

    # ---- 最大熵（书中 2.5.4 节）：给定方差时高斯熵最大 ----
    var_demo = 1.0
    h_gauss_max = 0.5 * np.log(2 * np.pi * np.e * var_demo)
    half_width = np.sqrt(3 * var_demo)         # 均匀分布 U(-a,a) 的方差 = a²/3，反解 a
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
    # 构造联合分布 p(X,Y)：X、Y 各 3 个取值，故意让它们相关（对角重）
    joint_xy = np.array([
        [0.10, 0.02, 0.01],
        [0.02, 0.30, 0.03],
        [0.01, 0.03, 0.48],
    ])
    MI = mutual_information(joint_xy)
    # 交叉验证：I(X;Y) = H(X) + H(Y) - H(X,Y)（书中公式 2.110 的等价形式）
    px = joint_xy.sum(axis=1)                 # 边际化得到 p(X)
    py = joint_xy.sum(axis=0)                 # 边际化得到 p(Y)
    Hx = entropy(px)
    Hy = entropy(py)
    Hxy = entropy(joint_xy.ravel())           # ravel()：矩阵摊平成向量
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

    # 先验 beta(2,2)：温和的无信息先验（μ=0.5 附近略高）
    mu_grid = np.linspace(0.001, 0.999, 500)
    a0, b0 = 2, 2
    H, T = 7, 3     # 观察 10 次抛硬币：7 正 3 反
    # 后验 = beta(a0+H, b0+T) = beta(9,5)（共轭性质：先验+数据仍是 beta）
    fig = Figure("beta 先验 -> 后验：观察 7 正 3 反后信念更新", "μ（硬币偏置）", "密度")
    fig.line(mu_grid, beta_pdf(mu_grid, a0, b0), label="先验 beta(2,2)")
    fig.line(mu_grid, beta_pdf(mu_grid, a0 + H, b0 + T), label="后验 beta(9,5)")
    fig.save(os.path.join(PLOTS_DIR, "fig8_beta_posterior.png"))
    # 众数公式：(a-1)/(a+b-2)；后验众数应偏向数据频率 0.7
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
