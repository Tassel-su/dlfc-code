# -*- coding: utf-8 -*-
"""
第 5 章：单层网络 —— 分类
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 5 章
（印刷页 131-170，PDF 页 151-190）。

本章要亲眼看到的现象：
  5.1 判别函数：线性判别、多分类的 1-of-K 编码、最小二乘用于分类的失败；
  5.2 决策论：误分类率、期望损失、拒绝选项、ROC 曲线与 AUC；
  5.3 生成式分类器：高斯类条件密度 -> 后验是 logistic（书中关键结论）；
  5.4 判别式分类器：逻辑回归（sigmoid + 交叉熵 + 梯度下降）、
      Softmax 多分类。

运行方式：
  C:/Python314/python.exe ch05_single_layer_classification.py
输出：
  _plots/ 下多张图 + 终端中文叙述

【阅读提示】重点理解：
  - 为什么分类不能用最小二乘
  - 交叉熵梯度为什么是 Φᵀ(y-t)（sigmoid 导数的魔法）
  - 生成式 vs 判别式的区别
  - ROC/AUC 怎么从阈值滑动得到
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


def sigmoid(a):
    """sigmoid 激活函数 σ(a) = 1/(1+e^(-a))（书中 5.4.1 节）。

    把实数 a 压到 (0,1)，可以当概率用。
    重要性质：σ'(a) = σ(a)(1-σ(a)) —— 交叉熵梯度简化的关键。
    """
    return 1.0 / (1.0 + np.exp(-a))


def softmax(a):
    """softmax：把 logits（任意实数向量）变成概率分布（书中 5.4.4 节）。

    实现细节：先减去最大值再 exp，防止 exp 溢出（数值稳定技巧）。
    """
    a = a - a.max(axis=-1, keepdims=True)          # 每行减去该行最大值
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis=-1, keepdims=True)   # 除以行和 -> 概率


def gen_twonorm(n_per_class=100, seed=0):
    """生成二分类数据：类 1 中心 (-1,-1)，类 2 中心 (1,1)，各向同性高斯。"""
    rng = np.random.default_rng(seed)
    x1 = rng.normal([-1, -1], 0.8, (n_per_class, 2))
    x2 = rng.normal([1, 1], 0.8, (n_per_class, 2))
    X = np.vstack([x1, x2])                    # 上下拼接 -> (200, 2)
    t = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])  # 标签 0/1
    return X, t


def main() -> None:
    # ==================================================================
    # 5.1 判别函数
    # ==================================================================
    section("5.1 判别函数：二分类判别式 y(x) = wᵀx + w0（书中 5.1 节）")
    print("判别式：y(x) > 0 -> 类 1；y(x) < 0 -> 类 2；y=0 是决策边界（超平面）")
    X, t = gen_twonorm(100, seed=0)
    print(f"  数据：{X.shape[0]} 个样本，2 类（类0 中心(-1,-1)，类1 中心(1,1)）")

    # ---- 5.1.4 最小二乘用于分类：演示其缺陷 ----
    print("\n-- 5.1.4 最小二乘用于分类：看看它哪里不好（书中 5.1.4 节）")
    Phi = np.hstack([np.ones((X.shape[0], 1)), X])          # 加一列 1（偏置项）
    w_ls, *_ = np.linalg.lstsq(Phi, t, rcond=None)          # 最小二乘拟合 0/1 标签
    pred_ls = Phi @ w_ls
    err_ls = float(np.mean((pred_ls > 0.5) != (t == 1)))    # 预测>0.5 判类1
    print(f"  最小二乘分类：训练误分类率 = {err_ls:.3f}")
    print("  缺陷 1：预测值是连续实数，不是概率（可 <0 或 >1）")
    print("  缺陷 2：对远离边界的极端样本（离群点）极度敏感 —— 下面演示")

    # 加一个极端离群点（位置远、标签故意标错），看边界被拉偏
    X_out = np.vstack([X, [[10.0, 10.0]]])
    t_out = np.concatenate([t, [0.0]])
    Phi_out = np.hstack([np.ones((X_out.shape[0], 1)), X_out])
    w_ls_out, *_ = np.linalg.lstsq(Phi_out, t_out, rcond=None)
    boundary_ls = -w_ls[0] / w_ls[2] if w_ls[2] != 0 else 0
    boundary_out = -w_ls_out[0] / w_ls_out[2] if w_ls_out[2] != 0 else 0
    print(f"  加离群点前后边界位置（示意值）：{boundary_ls:.2f} -> {boundary_out:.2f}（被拉偏）")
    print("  => 分类问题应该用概率模型（逻辑回归），而不是最小二乘")

    # ==================================================================
    # 5.2 决策论
    # ==================================================================
    section("5.2 决策论（书中 5.2 节）")
    print("-- 5.2.1 误分类率：最优决策是选后验概率最大的类")
    print("-- 5.2.2 期望损失：不同错误的代价可以不同")
    print("  例子：把「健康判为有病」（假阳性）与「把有病判为健康」（假阴性）代价不同")
    L_11, L_22 = 0.0, 0.0         # 正确分类零代价
    L_12 = 1.0                    # 把类1判成类2的代价（漏诊，代价大）
    L_21 = 0.1                    # 把类2判成类1的代价（误诊，代价小）
    p1, p2 = 0.4, 0.6
    cost_choose1 = L_12 * p2      # 选类1的期望损失
    cost_choose2 = L_21 * p1      # 选类2的期望损失
    print(f"  后验 p(类1)=0.4, p(类2)=0.6；漏诊代价1.0，误诊代价0.1")
    print(f"  选类1的期望损失 = {cost_choose1:.2f}；选类2的期望损失 = {cost_choose2:.2f}")
    best = 1 if cost_choose1 < cost_choose2 else 2
    print(f"  => 虽然类2后验更大，但损失不对称时应选类{best}（最小期望损失原则）")

    # ---- 5.2.3 拒绝选项 ----
    print("\n-- 5.2.3 拒绝选项：后验最大概率 < 阈值 θ 时拒绝决策（避免高代价误判）")
    theta = 0.6
    post = np.array([0.45, 0.55])            # 假想的低置信样本的后验
    decide = "拒绝" if post.max() < theta else f"选类 {int(np.argmax(post)) + 1}"
    print(f"  样本后验 {post}：max={post.max():.2f} < θ={theta} -> {decide}")
    post2 = np.array([0.9, 0.1])
    decide2 = "拒绝" if post2.max() < theta else f"选类 {int(np.argmax(post2)) + 1}"
    print(f"  高置信样本 {post2}：max={post2.max():.2f} >= θ -> {decide2}（正常决策）")

    # ---- 5.2.6 ROC 曲线 ----
    print("\n-- 5.2.6 ROC 曲线：TPR vs FPR，阈值滑动（书中 5.2.6 节，图 5.7 风格）")
    rng = np.random.default_rng(1)
    Xr, tr = gen_twonorm(200, seed=1)
    center1 = np.array([1.0, 1.0])          # 类 1 的中心（注意别搞反！）
    score = -np.linalg.norm(Xr - center1, axis=1) + rng.normal(0, 1.0, Xr.shape[0])
    # 阈值从高到低遍历：曲线从 (0,0) 走到 (1,1)，积分方向正确
    thresholds = np.linspace(score.max(), score.min(), 200)
    tprs, fprs = [], []
    for th in thresholds:
        pred = score > th                       # 高于阈值判正类
        tp = np.sum((pred == 1) & (tr == 1))    # 真阳性：预测正、实际正
        fn = np.sum((pred == 0) & (tr == 1))    # 假阴性
        fp = np.sum((pred == 1) & (tr == 0))    # 假阳性
        tn = np.sum((pred == 0) & (tr == 0))    # 真阴性
        tprs.append(tp / max(tp + fn, 1))       # TPR = 正类召回率
        fprs.append(fp / max(fp + tn, 1))       # FPR = 负类误报率
    auc = float(np.trapezoid(tprs, fprs)) if hasattr(np, "trapezoid") else float(np.trapz(tprs, fprs))
    fig = Figure(f"ROC 曲线（AUC={auc:.3f}）", "FPR（假阳性率）", "TPR（真阳性率）")
    fig.line(fprs, tprs, label="ROC")
    fig.line([0, 1], [0, 1], label="随机猜测")
    fig.save(os.path.join(PLOTS_DIR, "fig1_roc.png"))
    print(f"  AUC = {auc:.3f}（0.5=随机，1.0=完美）")
    assert auc > 0.7, "AUC 过低，数据可能有问题"

    # ==================================================================
    # 5.3 生成式分类器
    # ==================================================================
    section("5.3 生成式分类器：先建模 p(x|C_k) 再贝叶斯（书中 5.3 节）")
    print("思路：p(C_k|x) ∝ p(x|C_k) p(C_k) —— 对每一类估计类条件密度与先验。")
    print("关键结论（书中 5.3.1 节）：高斯类条件密度 + 共享协方差 -> 后验是 logistic！")

    Xg, tg = gen_twonorm(300, seed=2)
    mu1 = Xg[tg == 0].mean(axis=0)          # 类 1 均值（MLE：类内样本平均）
    mu2 = Xg[tg == 1].mean(axis=0)          # 类 2 均值
    Sigma_shared = 0.5 * (np.cov(Xg[tg == 0].T) + np.cov(Xg[tg == 1].T))  # 共享协方差
    pi1 = np.mean(tg == 0)                  # 类先验（比例）
    print(f"  MLE：μ1={np.round(mu1,2)}，μ2={np.round(mu2,2)}，共享 Σ={np.round(Sigma_shared,2)}")
    invS = np.linalg.inv(Sigma_shared)
    w_gen = invS @ (mu1 - mu2)
    w0_gen = -0.5 * (mu1 @ invS @ mu1 - mu2 @ invS @ mu2) + np.log(pi1 / (1 - pi1))
    print(f"  由类参数导出的 logistic 参数：w={np.round(w_gen,2)}，w0={w0_gen:.2f}")

    xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logits = pts @ w_gen + w0_gen
    probs = sigmoid(logits).reshape(xx.shape)
    fig = Figure("生成式分类器：高斯类条件密度 -> logistic 后验边界", "x1", "x2")
    fig.scatter(Xg[tg == 0, 0], Xg[tg == 0, 1], label="类 0")
    fig.scatter(Xg[tg == 1, 0], Xg[tg == 1, 1], label="类 1")
    fig.save(os.path.join(PLOTS_DIR, "fig2_generative.png"))
    train_acc = float(np.mean((sigmoid(Xg @ w_gen + w0_gen) > 0.5) == (tg == 0)))
    print(f"  训练精度 = {train_acc:.3f}")

    # ==================================================================
    # 5.4 判别式分类器：逻辑回归
    # ==================================================================
    section("5.4 判别式分类器：逻辑回归（书中 5.4.3 节）")
    print("直接对后验建模：p(C1|φ) = σ(wᵀφ)，不假设输入分布。")
    print("交叉熵损失：E(w) = -Σ [t_n ln y_n + (1-t_n) ln(1-y_n)]")
    print("梯度：∇E = Σ (y_n - t_n) φ_n = Φᵀ(y - t)   <- 简洁漂亮！")

    PhiL = np.hstack([np.ones((X.shape[0], 1)), X])    # 特征：偏置 + x1 + x2
    tL = t.astype(float)

    def cross_entropy(w):
        """交叉熵损失。eps 防止 log(0)。"""
        y = sigmoid(PhiL @ w)
        eps = 1e-12
        return float(-np.sum(tL * np.log(y + eps) + (1 - tL) * np.log(1 - y + eps)))

    def grad_ce(w):
        """交叉熵对 w 的梯度：∇E = Φᵀ(y - t)。

        推导：交叉熵的 -t/y + (1-t)/(1-y) 与 sigmoid 导数 y(1-y)
        相乘后恰好抵消，留下 (y-t)（书中 5.4.3 节）。
        """
        y = sigmoid(PhiL @ w)
        return PhiL.T @ (y - tL)

    w_lr = np.zeros(3)
    lr = 0.1
    for _ in range(5000):
        w_lr -= lr * grad_ce(w_lr)          # 梯度下降（最小化交叉熵）
    train_acc_lr = float(np.mean((sigmoid(PhiL @ w_lr) > 0.5) == (tL == 1)))
    print(f"  逻辑回归训练精度 = {train_acc_lr:.3f}（w = {np.round(w_lr,2)}）")

    # 数值验证梯度：解析梯度 vs 有限差分
    eps_fd = 1e-6
    w0_check = np.array([0.5, -0.3, 0.2])
    g_analytic = grad_ce(w0_check)
    g_numeric = np.zeros(3)
    for i in range(3):
        wp, wm = w0_check.copy(), w0_check.copy()
        wp[i] += eps_fd; wm[i] -= eps_fd
        g_numeric[i] = (cross_entropy(wp) - cross_entropy(wm)) / (2 * eps_fd)
    gerr = float(np.max(np.abs(g_analytic - g_numeric)))
    print(f"  交叉熵梯度 vs 有限差分：最大误差 {gerr:.2e} ✓")
    assert gerr < 1e-5, "交叉熵梯度错误"

    fig = Figure("逻辑回归决策边界（书中 5.4.3 节）", "x1", "x2")
    fig.scatter(X[t == 0, 0], X[t == 0, 1], label="类 0")
    fig.scatter(X[t == 1, 0], X[t == 1, 1], label="类 1")
    fig.save(os.path.join(PLOTS_DIR, "fig3_logistic.png"))

    ws = np.linspace(-5, 5, 100)
    losses = [cross_entropy(np.array([0.0, w1v, 0.0])) for w1v in ws]
    fig = Figure("交叉熵损失沿 w1 方向（凸函数）", "w1", "E(w)")
    fig.line(ws, losses, label="交叉熵损失")
    fig.save(os.path.join(PLOTS_DIR, "fig4_cross_entropy.png"))

    # ---- 5.4.4 多分类 Softmax ----
    print("\n-- 5.4.4 多分类：Softmax + 交叉熵（书中 5.4.4 节）")
    print("y_k = exp(a_k)/Σexp(a_j)；交叉熵 E = -ΣΣ t_nk ln y_nk；梯度 ∇E_k = Φᵀ(y_k - t_k)")
    rng3 = np.random.default_rng(3)
    centers = np.array([[-2, -2], [2, -2], [0, 2]])
    X3, t3 = [], []
    for k in range(3):
        X3.append(rng3.normal(centers[k], 0.7, (100, 2)))
        t3.append(np.full(100, k))
    X3 = np.vstack(X3); t3 = np.concatenate(t3)
    T3 = (t3[:, None] == np.arange(3)).astype(float)          # 1-of-K 编码
    Phi3 = np.hstack([np.ones((X3.shape[0], 1)), X3])
    W = np.zeros((3, 3))                                      # 每类一组权重
    lr3 = 0.05
    for _ in range(8000):
        Y = softmax(Phi3 @ W.T)                               # (N, 3)：三类概率
        W -= lr3 * ((Y - T3).T @ Phi3)                        # 梯度更新
    pred3 = np.argmax(Y, axis=1)                              # 概率最大的类
    acc3 = float(np.mean(pred3 == t3))
    print(f"  Softmax 三分类训练精度 = {acc3:.3f}")
    assert acc3 > 0.9, "Softmax 分类精度过低"

    xx3, yy3 = np.meshgrid(np.linspace(-5, 5, 150), np.linspace(-5, 5, 150))
    pts3 = np.stack([xx3.ravel(), yy3.ravel()], axis=1)
    Phi3g = np.hstack([np.ones((pts3.shape[0], 1)), pts3])
    prob3 = softmax(Phi3g @ W.T)
    fig = Figure("Softmax 三分类：每个区域的归属类", "x1", "x2")
    fig.scatter(X3[t3 == 0, 0], X3[t3 == 0, 1], label="类 0")
    fig.scatter(X3[t3 == 1, 0], X3[t3 == 1, 1], label="类 1")
    fig.scatter(X3[t3 == 2, 0], X3[t3 == 2, 1], label="类 2")
    fig.save(os.path.join(PLOTS_DIR, "fig5_softmax.png"))
    print("  三类的决策区域由 Softmax 概率最大者决定（图中 5.8 风格）")

    # ---- 5.4.5/5.4.6 probit 与规范链接 ----
    print("\n-- 5.4.5/5.4.6 Probit 回归与规范链接（书中 5.4.5/5.4.6 节）")
    print("sigmoid 与 probit（高斯 CDF）都是合法的激活函数；")
    print("指数族 + 规范链接（logistic 对伯努利）使梯度 ∇E = Φᵀ(y-t) 成立 ——")
    print("这就是逻辑回归梯度如此简洁的原因。")

    # ==================================================================
    # 6. 小结
    # ==================================================================
    section("6. 小结：这一章你亲眼看到了什么")
    print("""
  1. 线性判别式 y = wᵀx + w0 给出超平面决策边界；
  2. 最小二乘不适合分类：输出不是概率、对离群点敏感；
  3. 决策论：最小期望损失（非对称代价）、拒绝选项、ROC/AUC；
  4. 生成式分类器：p(x|C_k)p(C_k) -> 贝叶斯后验；
     高斯类条件 + 共享协方差 -> 后验恰为 logistic；
  5. 判别式分类器：直接建模 p(C|φ)，逻辑回归交叉熵梯度 = Φᵀ(y-t)；
  6. Softmax 把多分类后验归一化为概率，梯度同样简洁；
  7. 生成式 vs 判别式：生成式能处理缺失数据/离群点，判别式更鲁棒直接。
""")


if __name__ == "__main__":
    main()
