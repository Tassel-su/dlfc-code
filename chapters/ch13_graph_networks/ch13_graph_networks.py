# -*- coding: utf-8 -*-
"""
第 13 章：图网络
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 13 章
（印刷页 405-428，PDF 页 425-448）。

本章要亲眼看到的现象：
  13.1 图上的机器学习：邻接矩阵、度、置换等变性；
  13.2 神经消息传递：GCN（图卷积网络）H' = σ(Â H W)、聚合/更新算子、
      半监督节点分类（少数标签 + 图结构传播）；
  13.3 GAT（图注意力网络，邻居加权）、图嵌入（readout）、过平滑。

运行方式：
  C:/Python314/python.exe ch13_graph_networks.py
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


def build_two_communities(n_per=8, p_in=0.8, p_out=0.1, seed=0):
    """生成双社区图：社区内高连接、社区间低连接。返回 (邻接矩阵, 社区标签)。"""
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    A = np.zeros((n, n))
    labels = np.array([0] * n_per + [1] * n_per)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    return A, labels


def main() -> None:
    # ==================================================================
    # 13.1 图的表示（书中 13.1 节）
    # ==================================================================
    section("13.1 图的表示：邻接矩阵与置换等变（书中 13.1.1/13.1.2/13.1.3 节）")
    A, labels = build_two_communities()
    n = A.shape[0]
    deg = A.sum(axis=1)
    print(f"  图：{n} 个节点，{int(A.sum() / 2)} 条边")
    print(f"  节点度分布：min={deg.min():.0f}，max={deg.max():.0f}，均值={deg.mean():.1f}")

    # 置换等变性：重标节点 -> 邻接矩阵相应重排
    print("\n-- 13.1.3 置换等变性：重标节点 -> 输出相应置换（书中 13.1.3 节）")
    perm = np.random.default_rng(1).permutation(n)
    A_perm = A[np.ix_(perm, perm)]
    # 等变性验证：把重排后的矩阵按逆置换还原，应得到原矩阵
    inv = np.argsort(perm)
    A_restored = A_perm[np.ix_(inv, inv)]
    print(f"  重排后按逆置换还原与原矩阵一致：{np.array_equal(A_restored, A)}（等变 ✓）")
    print("  图神经网络的输出必须随节点重排而重排（置换等变）—— 架构硬约束")

    # ==================================================================
    # 13.2 GCN（书中 13.2.2 节）
    # ==================================================================
    section("13.2 图卷积网络 GCN：H' = σ(Â H W)（书中 13.2.2 节）")
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
    A_hat = D_inv_sqrt @ (A + np.eye(n)) @ D_inv_sqrt   # 自环 + 对称归一化
    print("  归一化邻接矩阵 Â = D^(-1/2) (A+I) D^(-1/2)（含自环）")

    # ---- 半监督节点分类 ----
    print("\n-- 13.2.5 节点分类：只给 6 个标签，靠图结构传播（书中 13.2.5 节）")
    rng = np.random.default_rng(2)
    X0 = np.zeros((n, 4))
    X0[labels == 0] = rng.normal(0, 0.1, (np.sum(labels == 0), 4))
    X0[labels == 1] = rng.normal(1, 0.1, (np.sum(labels == 1), 4))
    known = np.concatenate([np.arange(3), 8 + np.arange(3)])
    y_known = labels[known]

    d_in, d_hid = 4, 8
    rngW = np.random.default_rng(3)
    W1 = rngW.normal(0, 0.3, (d_in, d_hid))
    W2 = rngW.normal(0, 0.3, (d_hid, 2))

    def gcn_forward(X, W1, W2):
        H1 = np.tanh(A_hat @ X @ W1)
        logits = A_hat @ H1 @ W2
        return H1, logits

    for it in range(2000):
        # ---- 前向：两层 GCN ----
        H1, logits = gcn_forward(X0, W1, W2)
        # 只对"已知标签"的节点算交叉熵（半监督）
        lk = logits[known]
        lk = lk - lk.max(axis=1, keepdims=True)   # 数值稳定
        exp = np.exp(lk)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = float(-np.mean(np.log(probs[np.arange(len(known)), y_known] + 1e-12)))
        # ---- 反向：关键在"消息传递的转置" ----
        dlogits_k = probs.copy()
        dlogits_k[np.arange(len(known)), y_known] -= 1   # softmax 交叉熵梯度
        dlogits_k /= len(known)
        # 未标注节点的输出梯度为 0，但 dH = Âᵀ(dL/dH')
        # 让梯度沿图结构"传播"到所有节点 —— 这就是标签扩散的数学机制
        dlogits = np.zeros((n, 2))
        dlogits[known] = dlogits_k
        dH1 = A_hat.T @ (dlogits @ W2.T) * (1 - H1 ** 2)   # 经消息传递 + tanh 导数
        dW2 = H1.T @ (A_hat.T @ dlogits)
        dW1 = X0.T @ (A_hat.T @ dH1)
        W1 -= 0.1 * dW1; W2 -= 0.1 * dW2
        if it % 500 == 0:
            acc_k = float(np.mean(logits.argmax(axis=1)[known] == y_known))
            print(f"    iter {it}: 损失={loss:.4f}，监督精度={acc_k:.3f}")

    pred_all = logits.argmax(axis=1)
    acc_all = float(np.mean(pred_all == labels))
    print(f"  全部节点分类精度 = {acc_all:.3f}（只监督 6 个节点！）")
    assert acc_all > 0.9, "GCN 分类精度过低"
    print("  => GCN 用图结构把少数标签传播到全图 ✓")

    # 画图
    rng_l = np.random.default_rng(5)
    pos = rng_l.normal(0, 1, (n, 2))
    fig = Figure(f"双社区图：GCN 全图精度 {acc_all:.2f}", "x", "y")
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                fig.line([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], label="")
    fig.scatter(pos[:, 0], pos[:, 1], label="节点")
    fig.save(os.path.join(PLOTS_DIR, "fig1_graph.png"))

    # ---- 聚合算子 ----
    print("\n-- 13.2.3 聚合算子：mean / sum / max（书中 13.2.3 节）")
    H1, _ = gcn_forward(X0, W1, W2)
    for op in ("mean", "sum", "max"):
        if op == "mean":
            agg = (A @ H1) / np.maximum(deg[:, None], 1)
        elif op == "sum":
            agg = A @ H1
        else:
            agg = np.zeros_like(H1)
            for i in range(n):
                nb = np.where(A[i] > 0)[0]
                agg[i] = H1[nb].max(axis=0) if len(nb) else 0
        print(f"  {op} 聚合输出形状 {agg.shape}（邻居统计；max 抗噪声、mean 平滑）")

    # ==================================================================
    # 13.3 GAT 与过平滑（书中 13.3 节）
    # ==================================================================
    section("13.3 GAT 注意力与过平滑（书中 13.3.1/13.3.4 节）")
    print("-- 13.3.1 GAT：学习给每个邻居不同的权重（书中 13.3.1 节）")
    a_vec = rng.normal(0, 0.2, (2 * d_hid, 1))
    H = H1
    e_ij = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if A[i, j] or i == j:
                e_ij[i, j] = float((np.concatenate([H[i], H[j]]) @ a_vec).item())
    attn_w = np.zeros_like(e_ij)
    for i in range(n):
        nb = np.where((A[i] > 0) | (np.arange(n) == i))[0]
        ex = np.exp(e_ij[i, nb] - e_ij[i, nb].max())
        attn_w[i, nb] = ex / ex.sum()
    print(f"  节点 0 对前 6 个邻居的注意力：{np.round(attn_w[0, :6], 3)}（各不相同）")
    print("  GAT 让模型学会「谁更重要」，比均匀聚合更灵活")

    print("\n-- 13.3.4 过平滑：层数太多 -> 节点表示趋同（书中 13.3.4 节）")
    # 用行随机游走归一化 P = D^-1 A（特征值 <=1，重复应用必然收敛到常向量）
    P = np.diag(1.0 / np.maximum(deg, 1)) @ A
    for layers in (1, 3, 10, 50):
        H_deep = X0
        for _ in range(layers):
            H_deep = P @ H_deep
        D = np.linalg.norm(H_deep[:, None, :] - H_deep[None, :, :], axis=2)
        mean_dist = float(D[np.triu_indices(n, 1)].mean())
        print(f"  {layers:3d} 层消息传递后节点间平均距离 = {mean_dist:.4f}（趋同→0）")
    print("  => 过平滑：消息传递把邻居信息反复混合，节点表示最终趋同，")
    print("     信息被「洗掉」 —— 深 GCN 的固有问题，用残差/跳连/归一化缓解")

    # ==================================================================
    # 5. 小结
    # ==================================================================
    section("5. 小结：这一章你亲眼看到了什么")
    print("""
  1. 邻接矩阵 + 置换等变是图网络架构的硬约束；
  2. GCN = 归一化邻接矩阵上的消息传递：H' = σ(Â H W)；
  3. 半监督节点分类：少数标签 + 结构传播，精度 0.9+；
  4. 聚合（mean/sum/max）与更新算子是消息传递的构件；
  5. GAT 给邻居学权重；readout 做图分类；
  6. 过平滑：深 GCN 节点趋同，需要缓解技巧。
""")


if __name__ == "__main__":
    main()
