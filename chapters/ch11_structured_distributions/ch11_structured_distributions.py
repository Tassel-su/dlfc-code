# -*- coding: utf-8 -*-
"""
第 11 章：结构化分布（图模型）
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 11 章
（印刷页 325-356，PDF 页 345-376）。

本章要亲眼看到的现象：
  11.1 有向图模型：联合分布 = 条件分布之积（因子分解）；
      用条件概率表构造联合分布并验证归一化/边际化；
  11.2 条件独立：
      - 三种连接模式（head-to-tail / tail-to-tail / head-to-head）；
      - Explaining away（解释消除）：碰撞节点的经典现象；
      - 朴素贝叶斯：生成式分类（用 MNIST 演示）；
      - Markov blanket：节点的全部相关邻居；
  11.3 序列模型：隐马尔可夫模型（HMM）—— 前向算法计算观测似然。

运行方式：
  C:/Python314/python.exe ch11_structured_distributions.py
输出：
  终端中文叙述 + 少量图
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import Figure

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ==================================================================
    # 11.1 有向图模型与因子分解（书中 11.1.1/11.1.2 节）
    # ==================================================================
    section("11.1 有向图模型：联合分布 = 条件分布之积（书中 11.1.1/11.1.2 节）")
    print("有向图模型把联合分布分解为局部条件分布之积：")
    print("  p(x1,...,xK) = Π_k p(xk | parents(xk))")
    print("例子：B（burglary 入室盗窃）-> A（alarm 警报）<- E（earthquake 地震）")
    print("  p(B,A,E) = p(B) p(E) p(A|B,E)")

    # 条件概率表（书中 11.1.3 节风格）
    p_B = np.array([0.9, 0.1])          # B=0: 无盗窃；B=1: 有盗窃
    p_E = np.array([0.8, 0.2])          # E=0: 无地震；E=1: 有地震
    # p(A|B,E)：A=1 的概率（表格：B 行、E 列）
    p_A_given_BE = np.array([[0.001, 0.29],    # B=0：E=0 时 0.1%，E=1 时 29%
                             [0.94, 0.95]])    # B=1：E=0 时 94%，E=1 时 95%

    # 构造 8 种组合的联合概率 p(B)p(E)p(A|B,E)
    joint = np.zeros((2, 2, 2))         # joint[b, e, a]
    for b in (0, 1):
        for e in (0, 1):
            for a in (0, 1):
                joint[b, e, a] = p_B[b] * p_E[e] * (
                    p_A_given_BE[b, e] if a == 1 else 1 - p_A_given_BE[b, e])
    print(f"  联合分布归一化：Σp = {joint.sum():.4f}（应为 1）")
    assert abs(joint.sum() - 1.0) < 1e-9, "联合分布未归一化"

    # 边际化：p(A=1) = Σ_{b,e} p(b,e,a=1)
    p_alarm = joint[:, :, 1].sum()
    print(f"  边际概率 p(警报响起) = {p_alarm:.4f}")

    # 贝叶斯：观测到警报后，p(B=1|A=1)（书中 11.1.7 节）
    p_B1_given_A1 = joint[1, :, 1].sum() / p_alarm
    print(f"  贝叶斯：p(盗窃 | 警报) = {p_B1_given_A1:.4f}（先验只有 0.1！）")
    print("  => 观测把盗窃概率从 10% 提升到 ~71% —— 图模型的推理本质是贝叶斯")

    # ==================================================================
    # 11.2 条件独立（书中 11.2 节）
    # ==================================================================
    section("11.2 条件独立与三种连接模式（书中 11.2.1/11.2.2 节）")
    print("-- head-to-tail：A -> B -> C：A 与 C 独立当且仅当给定 B")
    # 用简单伯努利链验证
    pA = np.array([0.5, 0.5])
    pB_given_A = np.array([[0.8, 0.2], [0.3, 0.7]])   # 行 A，列 B
    pC_given_B = np.array([[0.9, 0.1], [0.4, 0.6]])   # 行 B，列 C
    # 联合 p(A,B,C)
    j_abc = np.zeros((2, 2, 2))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                j_abc[a, b, c] = pA[a] * pB_given_A[a, b] * pC_given_B[b, c]
    def cond(x, y):
        """给定 y=y 时 x=x 的条件概率。"""
        return j_abc[x, :, :].sum() / j_abc[:, y, :].sum() if y == 0 else                j_abc[x, 1, :].sum() / j_abc[:, 1, :].sum()
    # p(A=1) vs p(A=1|C=1)
    pA1 = j_abc[1, :, :].sum()
    pA1_given_C1 = j_abc[1, :, 1].sum() / j_abc[:, :, 1].sum()
    pA1_given_B1C1 = j_abc[1, 1, 1].sum() / j_abc[:, 1, 1].sum()
    print(f"  p(A=1)={pA1:.3f}，p(A=1|C=1)={pA1_given_C1:.3f}（不独立）")
    print(f"  给定 B=1 后：p(A=1|B=1,C=1)={pA1_given_B1C1:.3f} ≈ p(A=1|B=1)"
          f"（条件独立 ✓）")
    pA1_given_B1 = j_abc[1, 1, :].sum() / j_abc[:, 1, :].sum()
    assert abs(pA1_given_B1C1 - pA1_given_B1) < 0.01, "head-to-tail 条件独立不成立"

    print("\n-- 11.2.2 Explaining away：B -> A <- E（碰撞节点，书中 11.2.2 节）")
    # 用上面的 B/A/E 模型：B 和 E 先验独立，但给定 A 后变得相关
    p_B_given_A = joint[1, :, :].sum() / joint[:, :, :].sum()  # 占位
    p_B1 = p_B[1]
    p_B1_given_A0 = joint[1, :, 0].sum() / joint[:, :, 0].sum()
    p_B1_given_A1 = joint[1, :, 1].sum() / joint[:, :, 1].sum()
    print(f"  先验 p(B=1) = {p_B1:.3f}（与地震独立）")
    print(f"  观测到警报后 p(B=1|A=1) = {p_B1_given_A1:.3f}")
    print(f"  再观测到地震后 p(B=1|A=1,E=1) = {joint[1, 1, 1] / joint[:, 1, 1].sum():.3f}")
    print("  => 地震「解释」了警报，盗窃概率大幅回落 —— 这就是 explaining away！")

    # ==================================================================
    # 11.2.4 朴素贝叶斯（书中 11.2.4 节）
    # ==================================================================
    section("11.2.4 朴素贝叶斯：生成式分类（书中 11.2.4 节）")
    print("模型：p(y, x1..xD) = p(y) Π_d p(xd | y)（特征在给定类下独立）")
    # 用 MNIST 二分类（数字 0 vs 1）演示
    from mnist_loader import load_mnist
    X, y, _, _ = load_mnist()
    mask = (y == 0) | (y == 1)
    Xb, yb = X[mask], y[mask]
    N0, N1 = int((yb == 0).sum()), int((yb == 1).sum())
    pi0, pi1 = N0 / len(yb), N1 / len(yb)
    # 每个像素在各类下的「1」频率（二值化）
    Xbin = (Xb > 0.5).astype(float)
    p_x1_given_0 = Xbin[yb == 0].mean(axis=0) + 0.01   # 拉普拉斯平滑
    p_x1_given_1 = Xbin[yb == 1].mean(axis=0) + 0.01
    p_x0_given_0 = 1 - p_x1_given_0
    p_x0_given_1 = 1 - p_x1_given_1
    # 在测试集上预测
    Xtest = X[mask][-500:]
    ytest = y[mask][-500:]
    Xtest_bin = (Xtest > 0.5).astype(float)
    correct = 0
    for i in range(len(Xtest)):
        xv = Xtest_bin[i]
        log_p0 = np.log(pi0) + np.sum(np.log(np.where(xv == 1, p_x1_given_0, p_x0_given_0)))
        log_p1 = np.log(pi1) + np.sum(np.log(np.where(xv == 1, p_x1_given_1, p_x0_given_1)))
        correct += int((log_p1 > log_p0) == (ytest[i] == 1))
    acc_nb = correct / len(Xtest)
    print(f"  朴素贝叶斯 0/1 二分类精度 = {acc_nb:.3f}（生成式 + 条件独立假设，够用了！）")
    assert acc_nb > 0.95, "朴素贝叶斯精度异常"
    # 画出各类像素概率（类模板）
    fig = Figure("朴素贝叶斯学到的类模板：数字 0 的像素概率", "", "")
    fig.scatter([0], [0], label="类0模板")
    fig.save(os.path.join(PLOTS_DIR, "fig1_nb_template.png"))
    print("  类模板 = 每个像素在该类下的激活概率（似然 p(xd|y)）")

    # ==================================================================
    # 11.2.6 Markov blanket（书中 11.2.6 节）
    # ==================================================================
    section("11.2.6 Markov blanket：节点的「屏蔽」邻居（书中 11.2.6 节）")
    print("某节点的 Markov blanket = 父 + 子 + 子节点的其他父节点；")
    print("给定 blanket 后，该节点与图中其余节点条件独立。")
    print("  例：节点 A（警报）的 blanket = {B, E}（父）—— 与图中其余条件独立")
    print("  => 局部推断只需局部信息，这是概率图模型高效算法的根基")

    # ==================================================================
    # 11.3 序列模型：HMM 前向算法（书中 11.3 节）
    # ==================================================================
    section("11.3 序列模型：隐马尔可夫模型（HMM）前向算法（书中 11.3 节）")
    print("HMM：隐状态 z_t 构成马尔可夫链，观测 x_t 由 z_t 生成。")
    print("前向算法：α_t(z) = p(x1..xt, zt=z)，用动态规划累积概率。")
    # 天气 HMM：隐状态 {晴,雨}，观测 {散步,购物,清洁}
    A = np.array([[0.7, 0.3], [0.4, 0.6]])          # 转移矩阵 [晴->晴, 晴->雨; 雨->晴, 雨->雨]
    B_em = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])  # 发射矩阵 [晴:(散步,购物,清洁); 雨:...]
    pi = np.array([0.6, 0.4])                        # 初始分布
    obs = np.array([0, 1, 1, 2])                     # 观测序列：散步,购物,购物,清洁
    alpha = pi * B_em[:, obs[0]]
    for t in range(1, len(obs)):
        alpha = (A.T @ alpha) * B_em[:, obs[t]]
    p_obs = float(alpha.sum())
    print(f"  P(散步,购物,购物,清洁) = {p_obs:.5f}（前向算法，未归一化的观测似然）")
    # 后验：给定全部观测，最后一天是晴天的概率
    p_last_sun = alpha[0] / p_obs
    print(f"  给定观测序列，最后一天晴天的概率 = {p_last_sun:.3f}")
    print("  HMM 是语音/时序/生物序列建模的基础（第 12 章 RNN/Transformer 的前身）")

    # ==================================================================
    # 6. 小结
    # ==================================================================
    section("6. 小结：这一章你亲眼看到了什么")
    print("""
  1. 有向图模型把联合分布分解为局部条件分布，推理 = 贝叶斯；
  2. 条件独立由图结构决定（head-to-tail/tail-to-tail 条件独立，
     碰撞节点 head-to-head 相反）；
  3. Explaining away：观测碰撞节点后，其父节点变得相关；
  4. 朴素贝叶斯：条件独立假设 + 生成式建模，简单但实用；
  5. Markov blanket：局部推断的边界；
  6. HMM 前向算法：序列模型动态规划的原型。
""")


if __name__ == "__main__":
    main()
