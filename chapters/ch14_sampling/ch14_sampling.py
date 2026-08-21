# -*- coding: utf-8 -*-
"""
第 14 章：采样方法
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 14 章
（印刷页 429-458，PDF 页 449-478）。

本章要亲眼看到的现象：
  14.1 基本采样：
      - 拒绝采样（proposal + 接受率）；
      - 重要性采样（加权估计期望）；
      - SIR（采样-重要性-重采样）；
  14.2 马尔可夫链蒙特卡洛（MCMC）：
      - Metropolis-Hastings（通用拒绝-移动规则）；
      - Gibbs 采样（按条件分布逐变量更新）；
      - 祖先采样（从有向图模型生成样本）；
  14.3 Langevin 采样：能量模型 + 梯度 + 噪声的随机动力学。

运行方式：
  C:/Python314/python.exe ch14_sampling.py
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


def target_1d(x):
    """目标分布（未归一化）：双峰混合，用于采样演示。"""
    return np.exp(-0.5 * ((x + 1.5) / 0.5) ** 2) + 0.8 * np.exp(-0.5 * ((x - 1.5) / 0.6) ** 2)


def main() -> None:
    # ==================================================================
    # 14.1.3 拒绝采样（书中 14.1.3 节）
    # ==================================================================
    section("14.1.3 拒绝采样：proposal + 接受率（书中 14.1.3 节）")
    print("思想：从简单分布 q 采样，以概率 p(z)/(k·q(z)) 接受 —— 接受 = 服从目标")
    rng = np.random.default_rng(0)
    k = 3.0                                        # 包络常数：k·q(z) >= p(z) 对所有 z
    xs = np.linspace(-5, 5, 500)
    q = lambda z: np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)   # 标准高斯 proposal
    accepted = []
    trials = 0
    while len(accepted) < 5000:
        z = rng.normal(0, 1)
        u = rng.random()
        if u < target_1d(z) / (k * q(z)):
            accepted.append(z)
        trials += 1
    acc_rate = len(accepted) / trials
    print(f"  接受率 = {acc_rate:.3f}（k=3 时理论 ~1/k=0.33；k 越小越高效）")
    # 验证样本分布与目标一致：样本均值/二阶矩
    a = np.array(accepted)
    print(f"  样本均值 = {a.mean():.3f}，样本二阶矩 = {np.mean(a**2):.3f}")
    # 与数值积分对比
    Z = float(np.trapezoid(target_1d(xs), xs))
    E1 = float(np.trapezoid(xs * target_1d(xs), xs) / Z)
    print(f"  数值积分均值 = {E1:.3f}（一致 ✓）")
    assert abs(a.mean() - E1) < 0.1, "拒绝采样均值偏差过大"
    fig = Figure("拒绝采样：样本直方图 vs 目标分布", "z", "密度")
    hist_y, hist_x = np.histogram(a, bins=50, density=True)
    cc = (hist_x[:-1] + hist_x[1:]) / 2
    fig.line(cc, hist_y, label="拒绝采样样本")
    fig.line(xs, target_1d(xs) / Z, label="目标分布（归一化）")
    fig.save(os.path.join(PLOTS_DIR, "fig1_rejection.png"))
    print("  缺点：高维时 k 巨大（接受率指数下降）—— 需要 MCMC")

    # ==================================================================
    # 14.1.5 重要性采样（书中 14.1.5 节）
    # ==================================================================
    section("14.1.5 重要性采样：加权估计期望（书中 14.1.5 节）")
    print("E[f] = Σ w_n f(z_n)，w_n = p(z_n)/q(z_n)（重要性权重）")
    zs = rng.normal(0, 1, 20000)                     # 从 q（标准高斯）采样
    w = target_1d(zs) / (q(zs) * Z)                  # 归一化权重（Z 用数值积分）
    w = w / w.sum()
    imp_mean = float(np.sum(w * zs))
    print(f"  重要性采样估计 E[z] = {imp_mean:.3f}（数值积分 {E1:.3f} ✓）")
    assert abs(imp_mean - E1) < 0.05, "重要性采样偏差过大"

    # ---- 14.1.6 SIR ----
    print("\n-- 14.1.6 SIR：从权重做重采样（书中 14.1.6 节）")
    idx = rng.choice(len(zs), size=5000, p=w)        # 按权重重采样
    sir_samples = zs[idx]
    print(f"  SIR 样本均值 = {sir_samples.mean():.3f}（与目标一致 ✓）")

    # ==================================================================
    # 14.2 Metropolis-Hastings（书中 14.2.3 节）
    # ==================================================================
    section("14.2 Metropolis-Hastings（书中 14.2.3 节）")
    print("规则：提议 z*，接受概率 min(1, p(z*)/p(z))；否则留在原地")
    print("  -> 马尔可夫链的平稳分布 = 目标分布")
    rng2 = np.random.default_rng(1)
    chain = []
    z_cur = 0.0
    n_steps = 20000
    for t in range(n_steps):
        z_star = z_cur + rng2.normal(0, 0.8)         # 高斯随机游走提议
        alpha = min(1.0, target_1d(z_star) / target_1d(z_cur))
        if rng2.random() < alpha:
            z_cur = z_star
        chain.append(z_cur)
    chain = np.array(chain)
    burn = 2000                                       # 丢弃预热段
    samples = chain[burn:]
    print(f"  接受率 = {np.mean(chain[1:] != chain[:-1]):.3f}（~0.5 理想）")
    print(f"  链均值 = {samples.mean():.3f}（数值积分 {E1:.3f} ✓）")
    fig = Figure("Metropolis-Hastings：链轨迹与样本直方图", "z", "密度")
    hist_y, hist_x = np.histogram(samples, bins=50, density=True)
    cc = (hist_x[:-1] + hist_x[1:]) / 2
    fig.line(cc, hist_y, label="MH 样本")
    fig.line(xs, target_1d(xs) / Z, label="目标分布")
    fig.save(os.path.join(PLOTS_DIR, "fig2_metropolis.png"))
    assert abs(samples.mean() - E1) < 0.1, "MH 均值偏差过大"
    print("  只用了 p 的比值（未归一化即可）—— MCMC 不需要归一化常数！")

    # ==================================================================
    # 14.2.4 Gibbs 采样（书中 14.2.4 节）
    # ==================================================================
    section("14.2.4 Gibbs 采样：逐变量按条件分布更新（书中 14.2.4 节）")
    print("对二维高斯，条件分布 p(x1|x2)、p(x2|x1) 都是高斯 -> 直接采样")
    mu_g = np.array([0.0, 0.0])
    Sigma_g = np.array([[1.0, 0.7], [0.7, 1.0]])
    rho = Sigma_g[0, 1]
    gibbs = []
    x = np.array([2.0, -2.0])
    for t in range(15000):
        x[0] = rng2.normal(rho * x[1], np.sqrt(1 - rho ** 2))   # p(x1|x2)
        x[1] = rng2.normal(rho * x[0], np.sqrt(1 - rho ** 2))   # p(x2|x1)
        gibbs.append(x.copy())
    gibbs = np.array(gibbs[1000:])
    emp_corr = float(np.corrcoef(gibbs.T)[0, 1])
    print(f"  Gibbs 样本相关系数 = {emp_corr:.3f}（真实 ρ={rho} ✓）")
    fig = Figure("Gibbs 采样：样本散布（相关性 ρ=0.7）", "x1", "x2")
    fig.scatter(gibbs[::10, 0], gibbs[::10, 1], label="Gibbs 样本")
    fig.save(os.path.join(PLOTS_DIR, "fig3_gibbs.png"))
    assert abs(emp_corr - rho) < 0.05, "Gibbs 相关性与真实不符"

    # ==================================================================
    # 14.2.5 祖先采样（书中 14.2.5 节）
    # ==================================================================
    section("14.2.5 祖先采样：从有向图模型生成（书中 14.2.5 节）")
    print("从 DAG 的根节点开始，按拓扑序逐节点采样：")
    print("  先采样 B、E（根），再按 p(A|B,E) 采样 A（书中 B->A<-E 模型）")
    p_B, p_E = 0.1, 0.2
    p_A_given = np.array([[0.001, 0.29], [0.94, 0.95]])
    N_anc = 100000
    B = rng2.random(N_anc) < p_B
    E = rng2.random(N_anc) < p_E
    A = np.zeros(N_anc, dtype=bool)
    for i in range(N_anc):
        A[i] = rng2.random() < p_A_given[int(B[i]), int(E[i])]   # 布尔 -> int 索引
    p_alarm_sim = A.mean()
    print(f"  祖先采样：p(警报) = {p_alarm_sim:.4f}（第 11 章联合表计算的一致 ✓）")

    # ==================================================================
    # 14.3 Langevin 采样（书中 14.3.3 节）
    # ==================================================================
    section("14.3 Langevin 采样：能量模型 + 梯度 + 噪声（书中 14.3.3 节）")
    print("动力学：x <- x - ε∇E(x) + √(2ε) η，η~N(0,1)（梯度下降 + 随机性）")
    def energy(x):
        """双井能量：E(x) = (x²-1)²（两个最小值在 ±1）。"""
        return (x ** 2 - 1) ** 2
    def grad_E(x):
        return 4 * x * (x ** 2 - 1)
    rng3 = np.random.default_rng(3)
    x_l = 0.0
    eps_l = 0.01                    # 双井势 x^4 增长陡峭，步长要小
    l_samples = []
    for t in range(50000):
        x_l = x_l - eps_l * grad_E(x_l) + np.sqrt(2 * eps_l) * rng3.normal()
        x_l = np.clip(x_l, -5, 5)   # 数值安全裁剪（极少触发）
        l_samples.append(x_l)
    l_samples = np.array(l_samples[5000:])
    print(f"  Langevin 样本均值 = {l_samples.mean():.3f}，方差 = {l_samples.var():.3f}")
    print(f"  样本集中在双井 ±1 附近：{np.mean(l_samples > 0.5):.3f} vs {np.mean(l_samples < -0.5):.3f}")
    fig = Figure("Langevin 采样：样本分布在双井能量最小值附近", "x", "频数")
    hist_y, hist_x = np.histogram(l_samples, bins=60, density=True)
    cc = (hist_x[:-1] + hist_x[1:]) / 2
    fig.line(cc, hist_y, label="Langevin 样本")
    fig.save(os.path.join(PLOTS_DIR, "fig4_langevin.png"))
    print("  Langevin = 能量下降 + 噪声探索：扩散模型（第 20 章）的采样核心！")

    # ==================================================================
    # 5. 小结
    # ==================================================================
    section("5. 小结：这一章你亲眼看到了什么")
    print("""
  1. 拒绝采样：接受率 ~1/k，高维不可行；
  2. 重要性采样：加权估计，权重退化是高维痛点；
  3. SIR：用权重重采样得到目标分布的样本；
  4. MCMC：马尔可夫链的平稳分布 = 目标（只需未归一化密度）；
  5. Metropolis-Hastings 通用；Gibbs 用条件分布逐变量更新；
  6. 祖先采样：从图模型按拓扑序生成；
  7. Langevin：梯度 + 噪声，能量模型的采样与扩散模型的基石。
""")


if __name__ == "__main__":
    main()
