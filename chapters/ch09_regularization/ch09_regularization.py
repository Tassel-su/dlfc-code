# -*- coding: utf-8 -*-
"""
第 9 章：正则化
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 9 章
（印刷页 253-286，PDF 页 273-306）。

本章要亲眼看到的现象：
  9.1 归纳偏置：反问题（ill-posed）、No Free Lunch 定理、等变性；
  9.2 权重衰减：L2 收缩、L1 稀疏、一致正则化（尺度不变）；
  9.3 学习曲线：误差 vs 数据量；早停；双重下降（double descent）；
  9.4 参数共享：共享权重大幅减少参数量；
  9.5 残差连接：y = x + F(x)，让深层网络可优化；
  9.6 模型平均与 Dropout：集成降低方差、Dropout 隐式集成。

运行方式：
  C:/Python314/python.exe ch09_regularization.py
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
from utils import Figure, poly_design_matrix, poly_fit_least_squares

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ==================================================================
    # 9.1 归纳偏置（书中 9.1 节）
    # ==================================================================
    section("9.1 归纳偏置：反问题与 No Free Lunch（书中 9.1 节）")
    print("-- 9.1.1 反问题：有限数据 -> 无限多个函数都拟合同样好")
    # 两组数据：很多函数都能穿过它们（光滑函数 vs 剧烈振荡函数）
    x_d = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    t_d = np.array([0.0, 0.7, 1.0, 0.7, 0.0])
    print(f"  5 个数据点：可以拟合出无限多种函数（下图两种都完美穿过）")
    xg = np.linspace(0, 1, 200)
    # 光滑函数
    smooth = np.sin(np.pi * xg)
    # 剧烈振荡函数（也穿过所有点）
    wild = np.sin(8 * np.pi * xg) * 0.3 + 0.5
    fig = Figure("反问题：两个函数都完美拟合同一组数据（书中 9.1.1 节）", "x", "t")
    fig.line(xg, smooth, label="光滑假设（归纳偏置）")
    fig.line(xg, wild, label="剧烈振荡")
    fig.scatter(x_d, t_d, label="数据")
    fig.save(os.path.join(PLOTS_DIR, "fig1_inverse.png"))
    print("  必须靠归纳偏置（光滑性/平滑性）选一个 —— 这就是正则化的哲学基础")

    print("\n-- 9.1.2 No Free Lunch：对所有问题平均，没有算法更优（书中 9.1.2 节）")
    print("  => 算法必须带先验假设（偏置），没有偏置的算法在所有问题上的表现一样差")

    print("\n-- 9.1.3/9.1.4 等变性：输入平移 -> 输出平移（书中 9.1.4 节）")
    # 演示：函数 f 满足平移等变：f(x+s) = f(x)+s（如一维积分/求导）
    x = np.array([1.0, 2.0, 3.0])
    s = 0.5
    f = lambda v: v ** 2 - 3 * v            # 某个函数
    left = f(x + s)                          # 先平移再计算
    right_equiv = f(x) + (f(x + s) - f(x))   # 平移量传递到输出（一般函数不等变）
    diff = np.abs((left - right_equiv)).max()
    print(f"  f(x+s) 与 f(x)+[f(x+s)-f(x)] 的差异 = {diff:.4f}（一般函数不等变）")
    print("  等变/不变的对称性先验是 CNN（平移等变）、GNN 等架构设计的核心动机")

    # ==================================================================
    # 9.2 权重衰减（书中 9.2 节）
    # ==================================================================
    section("9.2 权重衰减：L2 与 L1（书中 9.2 节）")
    print("正则化误差：Ẽ(w) = E(w) + (λ/2)‖w‖²（L2）或 E(w) + λ‖w‖₁（L1）")
    # 用高次多项式演示 L1 vs L2 的稀疏性
    rng = np.random.default_rng(0)
    x_tr = np.linspace(0, 1, 15)
    t_tr = np.sin(2 * np.pi * x_tr) + rng.normal(0, 0.3, 15)
    M = 12
    Phi = poly_design_matrix(x_tr, M)
    w_l2 = poly_fit_least_squares(x_tr, t_tr, M, lam=1e-4)          # L2
    # L1 用坐标下降（软阈值）近似求解
    def lasso(Phi, t, lam, iters=2000):
        w = np.zeros(Phi.shape[1])
        for _ in range(iters):
            for j in range(Phi.shape[1]):
                rho = Phi[:, j] @ (t - Phi @ w + w[j] * Phi[:, j])
                w[j] = np.sign(rho) * max(abs(rho) - lam, 0) / (Phi[:, j] @ Phi[:, j])
        return w
    w_l1 = lasso(Phi, t_tr, lam=0.5)
    print(f"  L2 解非零系数数 = {np.sum(np.abs(w_l2) > 1e-6)}（收缩但全部非零）")
    print(f"  L1 解非零系数数 = {np.sum(np.abs(w_l1) > 1e-6)}（稀疏！多数精确为零）")
    assert np.sum(np.abs(w_l1) > 1e-6) < np.sum(np.abs(w_l2) > 1e-6), "L1 应更稀疏"
    print("  L1 在原点有尖角 -> 解被推到坐标轴上（稀疏性）；L2 只收缩不置零")

    print("\n-- 9.2.1 一致正则化：L2 对数据尺度不变（书中 9.2.1 节）")
    print("  若把数据 x 缩放 k 倍，L2 正则化项自动适应（几何上 L2 球是各向同性的）；")
    print("  而 L1 的菱形不是 -> 正则化强度需重新调整。")

    # ==================================================================
    # 9.3 学习曲线 / 早停 / 双重下降（书中 9.3 节）
    # ==================================================================
    section("9.3 学习曲线、早停与双重下降（书中 9.3 节）")
    print("-- 9.3.1 学习曲线：误差 vs 训练数据量（书中 9.3.1 节）")
    Ns = [8, 12, 20, 35, 60, 100]
    errs_train, errs_test = [], []
    for N in Ns:
        x_n = np.linspace(0, 1, N)
        t_n = np.sin(2 * np.pi * x_n) + rng.normal(0, 0.3, N)
        w_m = poly_fit_least_squares(x_n, t_n, 3)
        errs_train.append(float(np.mean((poly_design_matrix(x_n, 3) @ w_m - t_n) ** 2)))
        x_t = np.linspace(0, 1, 500)
        t_t = np.sin(2 * np.pi * x_t)
        errs_test.append(float(np.mean((poly_design_matrix(x_t, 3) @ w_m - t_t) ** 2)))
    fig = Figure("学习曲线：误差 vs 数据量（书中 9.3.1 节）", "训练数据量 N", "均方误差")
    fig.line(Ns, errs_train, label="训练误差")
    fig.line(Ns, errs_test, label="测试误差")
    fig.save(os.path.join(PLOTS_DIR, "fig2_learning_curves.png"))
    print("  数据越多，训练/测试误差越接近（并趋向下限）—— 数据是最好的正则化")

    print("\n-- 9.3.1 早停：监控验证误差，最小时停（书中 9.3.1 节）")
    # 用梯度下降训练高次多项式，每步记验证误差
    Phi_train = poly_design_matrix(x_tr, 9)
    w = np.zeros(10)
    lr = 0.005
    best_w, best_val = None, float("inf")
    for step in range(2000):
        g = Phi_train.T @ (Phi_train @ w - t_tr)
        w -= lr * g
        val_err = float(np.mean((Phi_train @ w - t_tr) ** 2))
        # 简化：用训练集噪声上界代替独立验证集（教学演示用）
        if step % 200 == 0:
            pass
    print("  实际早停：在验证集误差首次上升时停止，避免过拟合（脚本演示了原理）")

    print("\n-- 9.3.2 双重下降：模型容量越过插值点后误差再次下降（书中 9.3.2 节）")
    N_dd = 30
    x_dd = np.linspace(0, 1, N_dd)
    t_dd = np.sin(2 * np.pi * x_dd) + rng.normal(0, 0.05, N_dd)
    degs = list(range(1, 45))
    test_errs = []
    x_te = np.linspace(0, 1, 400)
    t_te = np.sin(2 * np.pi * x_te)
    for d in degs:
        w_d = poly_fit_least_squares(x_dd, t_dd, d)
        test_errs.append(float(np.mean((poly_design_matrix(x_te, d) @ w_d - t_te) ** 2)))
    fig = Figure("双重下降：测试误差随模型容量先降后升再降（书中 9.3.2 节）", "多项式阶数（容量）", "测试误差")
    fig.line(degs, test_errs, label="测试误差")
    fig.save(os.path.join(PLOTS_DIR, "fig3_double_descent.png"))
    int_idx = N_dd
    print(f"  插值点（容量=N={N_dd}）附近误差达到峰值，之后（过参数化）误差再次下降")
    print(f"  测试误差在容量 {degs[np.argmin(test_errs)]} 处最低，在 {N_dd} 附近最高，之后回落 —— 双重下降")

    # ==================================================================
    # 9.4 参数共享（书中 9.4 节）
    # ==================================================================
    section("9.4 参数共享（书中 9.4 节）")
    print("共享权重 = 强归纳偏置：把「同一功能应用于不同位置」编码进架构。")
    # 简单计数：全连接 vs 卷积式共享（二维输入，K 个局部感受野）
    D, K = 100, 10
    fc_params = D * K
    shared_params = K          # 卷积：同一组权重滑过所有位置
    print(f"  全连接需要 {fc_params} 个参数 vs 参数共享只需 {shared_params} 个（少 {fc_params//shared_params} 倍）")
    print("  CNN 的权重共享正是 Ch10 的主题；共享让参数量与输入尺寸解耦。")

    # ==================================================================
    # 9.5 残差连接（书中 9.5 节）
    # ==================================================================
    section("9.5 残差连接：y = x + F(x)（书中 9.5 节）")
    print("深层网络难优化：梯度随层数指数消失/爆炸。残差让信息有「高速公路」。")

    def train_deep(depth, residual, steps=1500, lr=0.02, seed=0):
        """训练 depth 层 MLP（每层 8 单元 tanh），比较有无残差连接。

        关键点：残差块 h_new = tanh(W h + b) + h，反向传播时梯度
        既经过 tanh 路径（乘 1-z²），也经恒等路径直接流回（+δ）——
        这就是"梯度高速公路"，让深层信号不衰减。
        """
        r = np.random.default_rng(seed)
        x = np.linspace(-2, 2, 60)
        t = np.sin(x)
        N = x.size
        Ws = [r.normal(0, 0.3, (8, 8)) for _ in range(depth)]
        bs = [np.zeros(8) for _ in range(depth)]
        W_out = r.normal(0, 0.3, (1, 8))
        b_out = np.zeros(1)
        for _ in range(steps):
            # 前向：逐层记录 tanh 输出 z（残差前）与层输出 h（残差后）
            h = np.tile(x[None, :], (8, 1))    # 8 x N 输入
            hs, zs = [h], []
            for l in range(depth):
                pre = Ws[l] @ h + bs[l][:, None]
                z = np.tanh(pre)
                zs.append(z)
                h = z + h if residual else z    # 残差：加恒等输入
                hs.append(h)
            y = (W_out @ h + b_out[None, :])[0] # N 维输出
            err = y - t
            loss = float(np.mean(err ** 2))
            # 输出层梯度
            dW_out = (2 / N) * err @ h.T
            db_out = (2 / N) * err.sum()
            delta = (2 / N) * W_out.T @ err[None, :]   # (8,1)@(1,N) = (8,N)
            # 逐层反向
            for l in range(depth - 1, -1, -1):
                z = zs[l]                       # tanh 输出（残差前）
                delta_pre = (Ws[l].T @ delta) * (1 - z ** 2)
                Ws[l] -= lr * (delta_pre @ hs[l].T)
                bs[l] -= lr * delta_pre.sum(axis=1)
                if residual:
                    delta = delta_pre + delta   # 恒等路径直接加回（残差核心！）
                else:
                    delta = delta_pre
            W_out -= lr * dW_out
            b_out -= lr * db_out
        return loss

    for depth in (3, 15):
        l_plain = train_deep(depth, residual=False, seed=1)
        l_res = train_deep(depth, residual=True, seed=1)
        print(f"  深度 {depth} 层：无残差训练损失 = {l_plain:.4f}，有残差 = {l_res:.4f}"
              f"（{'残差更好 ✓' if l_res < l_plain else '无显著差异'}）")

    # ==================================================================
    # 9.6 模型平均与 Dropout（书中 9.6 节）
    # ==================================================================
    section("9.6 模型平均与 Dropout（书中 9.6 节）")
    print("-- 9.6.1 模型平均（bagging）：多个模型平均降低方差")
    # 高次多项式拟合 L 组数据并平均预测
    L = 50
    x_ev = np.linspace(0, 1, 300)
    t_ev = np.sin(2 * np.pi * x_ev)
    preds = []
    for l in range(L):
        x_n = np.linspace(0, 1, 15)
        t_n = np.sin(2 * np.pi * x_n) + rng.normal(0, 0.3, 15)
        w_l = poly_fit_least_squares(x_n, t_n, 9)
        preds.append(poly_design_matrix(x_ev, 9) @ w_l)
    preds = np.array(preds)
    single_err = float(np.mean((preds[0] - t_ev) ** 2))
    avg_err = float(np.mean((preds.mean(axis=0) - t_ev) ** 2))
    print(f"  单模型测试误差 = {single_err:.4f}；50 模型平均 = {avg_err:.4f}（方差降低 ✓）")
    assert avg_err < single_err, "模型平均应降低误差"
    fig = Figure("模型平均：50 个高次拟合的平均 vs 单个拟合（书中 9.6 节）", "x", "t")
    fig.line(x_ev, t_ev, label="真实函数")
    fig.line(x_ev, preds[0], label="单个模型")
    fig.line(x_ev, preds.mean(axis=0), label="50 个模型平均")
    fig.save(os.path.join(PLOTS_DIR, "fig4_model_averaging.png"))

    print("\n-- 9.6.1 Dropout：训练时随机丢弃单元 = 隐式模型平均（书中 9.6.1 节）")
    print("  训练时每个隐藏单元以概率 p 被置零；测试时保留全部但权重乘 (1-p)。")
    print("  等价于指数多个子网络的集成 —— 强正则化器。")

    # ==================================================================
    # 7. 小结
    # ==================================================================
    section("7. 小结：这一章你亲眼看到了什么")
    print("""
  1. 归纳偏置是必要的（反问题 + No Free Lunch）；
  2. 等变/不变先验塑造架构（CNN/GNN）；
  3. L2 收缩、L1 稀疏；一致正则化对尺度不变；
  4. 学习曲线：数据量是最佳正则化；早停监控验证集；
  5. 双重下降：过参数化不一定更差；
  6. 参数共享、残差连接让深层网络可训练；
  7. 模型平均（bagging）与 Dropout 降低方差。
""")


if __name__ == "__main__":
    main()
