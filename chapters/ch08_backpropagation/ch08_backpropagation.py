# -*- coding: utf-8 -*-
"""
第 8 章：反向传播
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 8 章
（印刷页 233-252，PDF 页 253-272）。

本章要亲眼看到的现象：
  8.1 梯度计算：
      - 单层网络的梯度（闭式）；
      - 一般前馈网络的反向传播（链式法则）—— 用手推导一个 MLP；
      - 数值微分（有限差分）及其精度极限；
      - Jacobian 矩阵（向量值函数）与 Hessian 矩阵（二阶导数）；
  8.2 自动微分：
      - 前向模式（对偶数的方向导数）；
      - 反向模式（我们的 autograd 引擎）；
      - 两种模式的复杂度对比：反向一次遍历给全部参数梯度。

运行方式：
  C:/Python314/python.exe ch08_backpropagation.py
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
from autograd import Value
from utils import Figure

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ==================================================================
    # 8.1.1 单层网络的梯度（书中 8.1.1 节）
    # ==================================================================
    section("8.1.1 单层网络：梯度有闭式解（书中 8.1.1 节）")
    print("线性回归误差 E(w) = ½Σ(wᵀφ_n - t_n)²，梯度 ∇E = Φᵀ(Φw - t)")
    rng = np.random.default_rng(0)
    N, M = 20, 3
    Phi = rng.normal(0, 1, (N, M))
    t = rng.normal(0, 1, N)
    w_test = np.array([0.5, -0.3, 0.2])
    g_closed = Phi.T @ (Phi @ w_test - t)
    # 数值验证（中心差分）
    eps = 1e-6
    g_num = np.zeros(M)
    for i in range(M):
        wp, wm = w_test.copy(), w_test.copy()
        wp[i] += eps; wm[i] -= eps
        Ep = 0.5 * np.sum((Phi @ wp - t) ** 2)
        Em = 0.5 * np.sum((Phi @ wm - t) ** 2)
        g_num[i] = (Ep - Em) / (2 * eps)
    print(f"  闭式梯度 vs 中心差分：最大误差 {np.max(np.abs(g_closed - g_num)):.2e} ✓")

    # ==================================================================
    # 8.1.3 一般前馈网络：手推反向传播（书中 8.1.3 节）
    # ==================================================================
    section("8.1.3 手推反向传播：单隐层 MLP（书中 8.1.3 节）")
    print("网络：a = W1 x + b1；z = tanh(a)；y = W2 z + b2；E = ½(y-t)²")
    print("反向传播（链式法则，从输出往输入推）：")
    print("  dE/dy = y - t")
    print("  dE/dW2 = (y-t) zᵀ；dE/db2 = y - t")
    print("  dE/da = W2ᵀ(y-t) ⊙ (1-z²)   （tanh 导数）")
    print("  dE/dW1 = dE/da · xᵀ；dE/db1 = dE/da")

    # 具体数值例子：x=1 维输入，2 个隐藏单元
    x0, t0 = 0.5, 0.8
    W1 = np.array([[0.5], [-0.4]])
    b1 = np.array([0.1, 0.2])
    W2 = np.array([[0.7, 0.3]])
    b2 = np.array([-0.1])

    # 前向
    a1 = (W1 * x0).ravel() + b1          # W1(2x1) * 标量 x0 -> (2,) + b1(2,)
    z = np.tanh(a1)
    y = float(np.asarray(W2 @ z + b2).item())   # (1,) -> 标量
    E = 0.5 * (y - t0) ** 2
    print(f"  前向：a1={np.round(a1,4)}，z={np.round(z,4)}，y={y:.4f}，E={E:.4f}")
    # 反向（手推公式）
    dy = y - t0
    dW2 = dy * z
    db2 = dy
    da = (W2.ravel() * dy) * (1 - z ** 2)   # (2,)：逐元素，避免广播成矩阵
    dW1 = da * x0
    db1 = da
    print(f"  手推梯度：dE/dW2={np.round(dW2,4)}，dE/da={np.round(da.ravel(),4)}")

    # 用 autograd 引擎验证
    x_v = Value(x0, label="x")
    t_v = Value(t0, label="t")
    W1v = [Value(w, label=f"W1[{i}]") for i, w in enumerate(W1.ravel())]
    b1v = [Value(w, label=f"b1[{i}]") for i, w in enumerate(b1)]
    W2v = [Value(w, label=f"W2[{i}]") for i, w in enumerate(W2.ravel())]
    b2v = Value(b2[0], label="b2")
    # 前向计算图
    a = [W1v[0] * x_v + b1v[0], W1v[1] * x_v + b1v[1]]
    zz = [u.tanh() for u in a]
    yv = W2v[0] * zz[0] + W2v[1] * zz[1] + b2v
    Ev = (yv - t_v) ** 2 * 0.5
    Ev.backward()
    print(f"  autograd：dE/dW2={[round(w.grad, 4) for w in W2v]}（手推 {np.round(dW2,4)}）")
    print(f"  autograd：dE/db1={[round(w.grad, 4) for w in b1v]}（手推 {np.round(db1,4)}）")
    assert all(abs(w.grad - dW2[i]) < 1e-9 for i, w in enumerate(W2v)), "autograd 与手推不一致"
    print("  autograd 与手推一致 ✓")

    # ==================================================================
    # 8.1.4 数值微分（书中 8.1.4 节）
    # ==================================================================
    section("8.1.4 数值微分：有限差分的精度极限（书中 8.1.4 节）")
    # f(x) = sin(x) 在 x=1 的导数 = cos(1)
    f = lambda x: np.sin(x)
    true_deriv = np.cos(1.0)
    print(" eps      前向差分误差   中心差分误差")
    for eps in (1e-2, 1e-4, 1e-6, 1e-8, 1e-12):
        fwd = (f(1.0 + eps) - f(1.0)) / eps
        ctr = (f(1.0 + eps) - f(1.0 - eps)) / (2 * eps)
        print(f" {eps:.0e}    {abs(fwd-true_deriv):.3e}    {abs(ctr-true_deriv):.3e}")
    print("  中心差分精度高一个量级，但 eps 太小（1e-12）会因浮点舍入而退化")
    print("  => 数值微分只用于验证，实际训练用解析梯度/自动微分")

    # ==================================================================
    # 8.1.5 Jacobian 矩阵（书中 8.1.5 节）
    # ==================================================================
    section("8.1.5 Jacobian：向量值函数的导数（书中 8.1.5 节）")
    print("softmax: f(x) = [e^x1/Σe^x, ...]，Jacobian J_ij = ∂f_i/∂x_j")

    def softmax_np(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    x_j = np.array([1.0, 2.0, 0.5])
    # Jacobian 用前向模式（对偶）或数值：这里用解析公式 J = diag(p) - ppᵀ
    p = softmax_np(x_j)
    J_analytic = np.diag(p) - np.outer(p, p)
    # 数值验证
    J_num = np.zeros((3, 3))
    eps = 1e-6
    for j in range(3):
        xp = x_j.copy(); xp[j] += eps
        xm = x_j.copy(); xm[j] -= eps
        J_num[:, j] = (softmax_np(xp) - softmax_np(xm)) / (2 * eps)
    err = float(np.abs(J_analytic - J_num).max())
    print(f"  softmax Jacobian 解析公式 vs 数值：最大误差 {err:.2e} ✓")
    print(f"  J 的行和 = {np.round(J_analytic.sum(axis=1), 6)}（应为 0：概率和不变）")

    # ==================================================================
    # 8.1.6 Hessian 矩阵（书中 8.1.6 节）
    # ==================================================================
    section("8.1.6 Hessian：二阶导数矩阵（书中 8.1.6 节）")
    print("二次误差 E(w) = ½(w-w*)ᵀH(w-w*) 的 Hessian 就是 H（常数矩阵）")
    H = np.array([[2.0, 0.5], [0.5, 1.0]])
    print(f"  E(w) 的 Hessian = {np.round(H, 2)}（特征值 {np.round(np.linalg.eigvalsh(H),3)}）")
    print("  Hessian 特征值 = 各方向曲率：用于牛顿法、曲率诊断（第 7 章条件数的来源）")

    # ==================================================================
    # 8.2 自动微分：前向模式 vs 反向模式（书中 8.2 节）
    # ==================================================================
    section("8.2 自动微分（书中 8.2 节）")

    # ---- 8.2.1 前向模式：对偶数 ----
    print("\n-- 8.2.1 前向模式：对偶数（dual numbers）")
    print("把 x 替换为 x + ε，计算中 ε²=0，则结果的 ε 系数 = 方向导数")
    # 演示：f(x) = x² + 3x，在 x=2 沿方向 d=1 的导数 = 2x+3 = 7
    def dual(f, x, d=1.0):
        """前向模式自动微分：f 接受 (x, dx) 返回 (y, dy)。"""
        return f(x, d)
    def f_dual(x, d):
        # x² + 3x：dx = 1, d(x²)=2x·dx, d(3x)=3dx
        return x * x + 3 * x, 2 * x * d + 3 * d
    y_val, dy = f_dual(2.0, 1.0)
    print(f"  f(2)={y_val}，前向模式导数 = {dy}（解析 2·2+3=7 ✓）")

    # ---- 8.2.2 反向模式：复杂度对比 ----
    print("\n-- 8.2.2 反向模式 vs 前向模式：复杂度（书中 8.2.1/8.2.2 节）")
    print("  对 F: R^D -> R^M（D 输入，M 输出）：")
    print("    前向模式：D 次遍历（每输入一次）—— D 大时昂贵")
    print("    反向模式：M 次遍历（每输出一次）—— 深度网络 M=1（损失是标量）！")
    print("  损失函数只有 1 个输出 -> 反向模式一次遍历得到全部参数梯度")
    print("  => 这就是深度学习框架都用反向传播的原因")

    # 用 autograd 引擎验证一个稍大的 MLP（2-3-1），与数值梯度对比
    print("\n  用 autograd 引擎训练小型 MLP 并全程验证梯度（2→3→1 网络）：")
    rng2 = np.random.default_rng(1)
    W1p = [Value(float(v)) for v in rng2.normal(0, 1, 6)]     # 2x3
    b1p = [Value(float(v)) for v in rng2.normal(0, 1, 3)]
    W2p = [Value(float(v)) for v in rng2.normal(0, 1, 3)]     # 3x1
    b2p = Value(float(rng2.normal(0, 1)))

    def mlp_loss(x1, x2, tt):
        """用 autograd 计算 2-3-1 MLP 的损失。"""
        h = [(W1p[0] * x1 + W1p[1] * x2 + b1p[0]).tanh(),
             (W1p[2] * x1 + W1p[3] * x2 + b1p[1]).tanh(),
             (W1p[4] * x1 + W1p[5] * x2 + b1p[2]).tanh()]
        y = W2p[0] * h[0] + W2p[1] * h[1] + W2p[2] * h[2] + b2p
        return (y - tt) ** 2 * 0.5

    Xd = rng2.normal(0, 1, (10, 2))
    Td = rng2.normal(0, 1, 10)

    def compute_total():
        """重建计算图并返回总损失（每次都要新建图！）。"""
        tot = None
        for xi, ti in zip(Xd, Td):
            li = mlp_loss(Value(float(xi[0])), Value(float(xi[1])), Value(float(ti)))
            tot = li if tot is None else tot + li
        return tot

    total = compute_total()
    total.backward()
    params = W1p + b1p + W2p + [b2p]
    g_auto = np.array([p.grad for p in params])
    # 数值梯度（中心差分）：改参数后必须【重建计算图】再取值
    g_num = np.zeros_like(g_auto)
    eps = 1e-6
    for i in range(len(params)):
        orig = params[i].data
        params[i].data = orig + eps
        E1 = compute_total().data
        params[i].data = orig - eps
        E2 = compute_total().data
        params[i].data = orig
        g_num[i] = (E1 - E2) / (2 * eps)
    err_mlp = float(np.max(np.abs(g_auto - g_num)))
    print(f"  autograd 全部 {len(params)} 个参数梯度 vs 数值：最大误差 {err_mlp:.2e} ✓")
    assert err_mlp < 1e-5, "autograd 与数值梯度不一致"

    # ==================================================================
    # 4. 小结
    # ==================================================================
    section("4. 小结：这一章你亲眼看到了什么")
    print("""
  1. 反向传播 = 链式法则的系统化应用：从输出往输入逐层传梯度；
  2. 数值微分精度有限（中心差分好于前向，但 eps 太小会退化），
     只用于验证解析梯度；
  3. Jacobian（一阶矩阵）与 Hessian（二阶矩阵）描述向量值函数的曲率；
  4. 自动微分把"求导"变成程序化操作：
     前向模式 D 次遍历，反向模式 M 次遍历；
     损失只有一个输出（M=1）-> 反向模式一次遍历得全部梯度；
  5. 我们的 autograd 引擎（autograd.py）就是 PyTorch 的微型版，
     后续章节（Transformer/VAE/扩散）全部用它自动求梯度。
""")


if __name__ == "__main__":
    main()
