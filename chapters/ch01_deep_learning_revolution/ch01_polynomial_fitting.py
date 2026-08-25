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

【阅读提示】本脚本被逐行注释，重点看：
  - numpy 的"向量化"写法（没有 for 循环，一次算整批）
  - 设计矩阵 Phi 是什么、为什么能用闭式解
  - 数值梯度检验、留一法这些"验证"代码在干什么
"""
import os          # 操作系统接口：路径处理、文件操作
import sys         # Python 运行时：sys.path（模块搜索路径）

import numpy as np # numpy：全书唯一的数值计算库（数组、矩阵、随机数）

# ---------------------------------------------------------------------------
# 路径引导：让本脚本能找到仓库根目录里的 utils.py
# ---------------------------------------------------------------------------
# __file__ 是"本脚本自己的完整路径"，形如：
#   C:/Users/eric/Desktop/dlfc-code/chapters/ch01_xxx/ch01_polynomial_fitting.py
# os.path.abspath(__file__)  -> 转成绝对路径（保险）
# os.path.dirname(...)       -> 去掉文件名，向上取一层目录
# 脚本位于 repo/chapters/chXX/ 下，所以要往上取 3 层才到仓库根目录：
#   ch01_xxx/ -> chapters/ -> repo/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path 是 Python 找模块的搜索路径列表。insert(0, ...) 把仓库根放到最前面，
# 这样下面 "from utils import ..." 才能找到仓库根目录下的 utils.py。
sys.path.insert(0, _REPO_ROOT)
# 从共享工具模块导入本章要用的函数（每个函数都在 utils.py 里有逐行注释）
from utils import (Figure, check_gradient, gen_sin_data, poly_design_matrix,
                   poly_eval, poly_fit_least_squares, rms_error)

# 图输出目录：本脚本所在目录下的 _plots 子目录
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_plots")


def section(title: str) -> None:
    """打印小节标题分隔线，方便在终端里看清当前跑到哪一节。

    print("=" * 68) 是"字符串 * 数字"：把 "=" 重复 68 次，纯装饰。
    """
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. 数据生成（书中 1.2 节）
    # ------------------------------------------------------------------
    section("1. 数据生成：带噪的 sin(2πx)（书中 1.2 节，PDF p.4）")
    N = 10                 # 训练样本数（与书中一致：只有 10 个点！）
    sigma = 0.3            # 噪声标准差（与书中一致）
    # gen_sin_data 是 utils.py 里的函数：
    #   x = [0, 1/9, 2/9, ..., 1]（10 个点均匀分布在 [0,1]）
    #   t = sin(2πx) + 高斯噪声
    # seed=42 让随机数固定：每次运行都得到完全相同的数据（可复现）
    x_train, t_train = gen_sin_data(N=N, sigma=sigma, seed=42)
    print(f"训练数据：{N} 个点，x ∈ [0,1] 均匀分布，t = sin(2πx) + N(0, {sigma}²)")
    print(f"x = {np.round(x_train, 3)}")   # np.round(..., 3) 保留 3 位小数，打印好看
    print(f"t = {np.round(t_train, 3)}")

    # 测试数据：为了评估"泛化能力"，准备两套东西——
    #  (a) 高密度的"真实函数"曲线（无噪声），画图用
    x_true = np.linspace(0, 1, 500)        # 500 个点，比训练数据密得多
    t_true = np.sin(2 * np.pi * x_true)    # 真实函数（无噪声）
    #  (b) 干净的测试集：1000 个点，也是带噪的，用来算"测试误差"
    rng = np.random.default_rng(123)       # 另一个种子，保证测试集也可复现
    x_test = np.linspace(0, 1, 1000)
    t_test = np.sin(2 * np.pi * x_test) + rng.normal(0.0, sigma, size=x_test.size)
    # 注意 rng.normal(0.0, sigma, size=...) 的第三个参数：
    #   生成与 x_test 同样数量的独立高斯噪声

    # 画第 1 张图：真实函数曲线 + 10 个带噪训练点
    # Figure 是 utils.py 的绘图类：先建画布，再画线/点，最后存图
    fig = Figure("第 1 章：训练数据（10 个带噪点）与真实函数 sin(2πx)", "x", "t")
    fig.line(x_true, t_true, label="真实函数 sin(2πx)")   # 蓝色曲线
    fig.scatter(x_train, t_train, label="训练数据")         # 散点（10 个点）
    fig.save(os.path.join(PLOTS_DIR, "fig1_sin_data.png")) # 存成 PNG

    # ------------------------------------------------------------------
    # 2. 多项式模型与最小二乘闭式解 + 两种验证
    # ------------------------------------------------------------------
    section("2. 多项式基函数与最小二乘正规方程（书中 1.2 节）")
    print("模型：y(x, w) = Σ_j w_j x^j，设计矩阵 Phi[n,j] = x_n^j")
    print("误差函数：E(w) = ½ Σ_n (y(x_n,w) - t_n)²  ->  闭式解 w* = (ΦᵀΦ)⁻¹ Φᵀ t")

    M = 9  # 先取一个高阶级数做验证（M=9 会过拟合，正是我们要观察的）

    # ---- 验证①：M=3 时，正规方程 vs numpy 的 lstsq ----
    # 设计矩阵：Phi[n,j] = (x_n)^j，即每行是 [1, x, x², ..., x^M]
    # 为什么需要它？把多项式写成矩阵形式 y = Phi @ w，
    # 这样"求系数"变成"解线性方程组"，一步算完。
    M_small = 3
    w_solve = poly_fit_least_squares(x_train, t_train, M_small, method="solve")
    Phi_small = poly_design_matrix(x_train, M_small)
    w_lstsq3, *_ = np.linalg.lstsq(Phi_small, t_train, rcond=None)
    diff_small = np.max(np.abs(w_solve - w_lstsq3))
    assert diff_small < 1e-9, f"M=3 正规方程与 lstsq 不一致：{diff_small}"
    print(f"验证① M=3 时正规方程解 vs lstsq：最大偏差 {diff_small:.2e} ✓（公式正确）")

    # ---- 验证①'：M=9 时 ΦᵀΦ 病态（教学点：公式正确但数值会崩）----
    Phi = poly_design_matrix(x_train, M)
    cond9 = np.linalg.cond(Phi.T @ Phi)
    w_solve9 = poly_fit_least_squares(x_train, t_train, M, method="solve")
    w_lstsq9, *_ = np.linalg.lstsq(Phi, t_train, rcond=None)
    diff9 = np.max(np.abs(w_solve9 - w_lstsq9))
    print(f"验证①' M=9 时 cond(ΦᵀΦ) ≈ {cond9:.2e}（病态！），"
          f"正规方程 vs lstsq 偏差 {diff9:.1f} —— 所以代码用 lstsq")
    print(f"       （书本公式 w*=(ΦᵀΦ)⁻¹Φᵀt 在理论上正确，数值上要交给 SVD）")

    # ---- 验证②：解析梯度 vs 数值差分（为第 8 章反向传播预热）----
    # 误差函数 E(w) = ½Σ(y_n - t_n)² 的梯度可以手推：
    #   ∇E = Φᵀ (Φw - t)    （对每个 w_j 求偏导后整理成矩阵形式）
    def error(w):
        # Phi @ w  : 计算所有样本的预测值（矩阵乘法，一次算 10 个点）
        # 减 t_train: 预测与真实的差
        # ** 2      : 逐元素平方
        # np.sum   : 求和；乘 0.5 是公式里的 ½（让梯度表达式更干净）
        return 0.5 * float(np.sum((Phi @ w - t_train) ** 2))

    def grad_error(w):
        # 解析梯度公式：∇E = Φᵀ (Φw - t)
        return Phi.T @ (Phi @ w - t_train)

    w0 = np.linspace(-2, 2, M + 1)   # 随便选一个检验点（一维数组，长度 M+1）
    # check_gradient（utils.py）：用中心差分 (f(x+h)-f(x-h))/2h 算数值梯度，
    # 再和解析梯度比较。这是"代码验证数学推导"的标准方法。
    max_diff, ok = check_gradient(error, grad_error, w0)
    assert ok, "梯度检验未通过"
    print(f"验证② E(w) 解析梯度 vs 有限差分：最大误差 {max_diff:.2e} ✓")

    # ------------------------------------------------------------------
    # 3. 不同阶数 M = 0..9 的拟合：欠拟合 vs 过拟合
    # ------------------------------------------------------------------
    section("3. 不同阶数的拟合效果：欠拟合 vs 过拟合（书中图 1.4/1.5）")
    M_list = list(range(0, 10))   # [0,1,2,...,9]：10 种模型容量
    rms_train, rms_test = [], []  # 两个空列表，分别存每个 M 的训练/测试 RMS
    fits = {}                     # 字典：M -> 系数 w（后面画图要用）
    for m in M_list:              # 循环每种阶数
        w_m = poly_fit_least_squares(x_train, t_train, m)   # 拟合该阶数
        fits[m] = w_m             # 存下系数
        # rms_error（utils.py）：均方根误差 sqrt(mean((预测-真实)²))
        rms_train.append(rms_error(t_train, poly_eval(x_train, w_m)))
        rms_test.append(rms_error(t_test, poly_eval(x_test, w_m)))
        # poly_eval(x, w)：用系数 w 计算多项式在 x 处的值（预测）

    # 打印表格：观察"训练 RMS 单调下降、测试 RMS 呈 U 形"
    print(" M | 训练 RMS | 测试 RMS   <- E_RMS = sqrt( 2E(w*)/N )，书中公式")
    print("---+" + "-" * 30)       # 表格分隔线（字符串拼接）
    for m, (r_tr, r_te) in zip(M_list, zip(rms_train, rms_test)):
        # zip(M_list, zip(rms_train, rms_test))：把三列并排
        print(f"{m:2d} |  {r_tr:.4f}  |  {r_te:.4f}")
        # f"{m:2d}" 宽度 2 右对齐；"{r_tr:.4f}" 保留 4 位小数

    # 图：四条典型拟合曲线（M=0,1,3,9）—— 直观看到欠拟合到过拟合
    fig = Figure("不同阶数多项式拟合（书中图 1.4 风格）", "x", "t")
    fig.line(x_true, t_true, label="真实函数")
    for m in (0, 1, 3, 9):        # 只画 4 条，避免图太乱
        fig.line(x_true, poly_eval(x_true, fits[m]), label=f"M={m}")
    fig.scatter(x_train, t_train, label="训练数据")
    fig.save(os.path.join(PLOTS_DIR, "fig2_poly_fits.png"))

    # 图：RMS 随 M 变化（书中图 1.5：测试误差 U 形曲线）
    fig = Figure("训练/测试 RMS 随阶数 M 变化（书中图 1.5）", "M", "E_RMS")
    fig.line(M_list, rms_train, label="训练 RMS")
    fig.line(M_list, rms_test, label="测试 RMS")
    fig.save(os.path.join(PLOTS_DIR, "fig3_rms_vs_M.png"))

    best_m_test = int(np.argmin(rms_test))   # 测试误差最小的 M（U 形最低点）
    print(f"测试集上最优 M = {best_m_test}（训练集上 RMS 随 M 单调下降，测试集呈 U 形 = 过拟合信号）")

    # ------------------------------------------------------------------
    # 4. L2 正则化：M=9 时用 λ 收缩系数
    # ------------------------------------------------------------------
    section("4. L2 正则化抑制过拟合（书中 1.2 节，PDF p.9-11）")
    print("正则化误差：Ẽ(w) = ½ Σ_n (y(x_n,w)-t_n)² + (λ/2)||w||²")
    print("闭式解：w* = (ΦᵀΦ + λI)⁻¹ Φᵀ t")
    print("书中用 M=9：ln λ = -∞（无正则）时振荡剧烈；ln λ = -18 时被拉回平滑")

    # 画出三种 λ 的拟合曲线对比
    fig = Figure("M=9 时 L2 正则化的效果（书中图 1.7 风格）", "x", "t")
    fig.line(x_true, t_true, label="真实函数")
    for ln_lam in (-np.inf, -18.0, 0.0):   # 三个正则强度
        # ln λ = -∞ 表示 λ=0（无正则）；np.exp(-18) 是很小的 λ；λ=e^0=1 很强
        lam = 0.0 if np.isneginf(ln_lam) else np.exp(ln_lam)
        w_reg = poly_fit_least_squares(x_train, t_train, M, lam=lam)
        fig.line(x_true, poly_eval(x_true, w_reg),
                 label="无正则 (lnλ=-∞)" if lam == 0 else f"ln λ = {ln_lam:.0f}")
    fig.scatter(x_train, t_train, label="训练数据")
    fig.save(os.path.join(PLOTS_DIR, "fig4_regularization.png"))

    # 打印系数：观察"正则化把巨大系数压小"（收缩效应）
    print("M=9 的系数 w_j（感受正则化的收缩效应）：")
    for ln_lam in (-np.inf, -18.0, 0.0):
        lam = 0.0 if np.isneginf(ln_lam) else np.exp(ln_lam)
        w_reg = poly_fit_least_squares(x_train, t_train, M, lam=lam)
        label = "无正则" if lam == 0 else f"lnλ={ln_lam:.0f}"
        print(f"  {label:10s}: {np.round(w_reg, 3)}")
        # 你会看到：无正则时系数高达 10^5，λ 变大后缩到 10^0 量级

    # ------------------------------------------------------------------
    # 5. 留一法 (LOO) 模型选择
    # ------------------------------------------------------------------
    section("5. 留一法模型选择：不碰测试集选出 M=3（书中 1.3 节）")
    print("LOO：对每个样本 n，用除 n 外的 N-1 个点训练，再预测第 n 个点，")
    print("累计平方误差 E_LOO(M) = Σ_n (t_n - y^(n)(x_n))²。")

    loo_errors = []                       # 每个 M 的 LOO 误差
    for m in M_list:
        err_sum = 0.0                     # 累加器
        for n in range(N):                # 对每个样本做"留一"
            # idx 是布尔数组：除了第 n 个都是 True
            # np.arange(N) 生成 [0,1,...,N-1]，!= n 得到 [True,False,...]
            idx = np.arange(N) != n
            # 只用"其余 N-1 个点"训练（布尔索引就是"挑出 True 位置"）
            w_m = poly_fit_least_squares(x_train[idx], t_train[idx], m)
            # 预测被留下的第 n 个点；np.array([x_train[n]]) 转成数组才能算
            err_sum += (t_train[n] - poly_eval(np.array([x_train[n]]), w_m)[0]) ** 2
        loo_errors.append(err_sum)        # 该 M 的总 LOO 误差
    best_m_loo = int(np.argmin(loo_errors))   # 误差最小的 M

    # 画 LOO 误差曲线
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
    # 只有"直接运行本文件"时才执行 main()；
    # 如果被 import 则不会执行（这是 Python 的标准入口写法）
    main()
