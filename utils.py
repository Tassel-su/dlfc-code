# -*- coding: utf-8 -*-
"""
dlfc-code 共享工具模块
======================
配套《Deep Learning: Foundations and Concepts》(Bishop & Bishop, Springer 2023)
逐章代码化的公共工具。全书 20 章的脚本都从这里 import 函数。

本文件包含 6 组工具：
  1. 绘图（Figure / SVGPlot / save_figure）
  2. 误差度量（rms_error）
  3. 多项式基函数与最小二乘（Ch1/Ch4 共用）
  4. 数值梯度检验（Ch8 反向传播的验证工具）
  5. 概率与信息论工具（Ch2/Ch3 共用）
  6. 第 1 章的合成数据生成

阅读建议：先看本文件的函数签名和注释，再去看各章脚本如何使用。
"""
from __future__ import annotations

import os
import sys

import numpy as np   # 全书所有数值计算的基础库（数组、矩阵运算、随机数）

# matplotlib 的字体/缓存默认写到用户目录（C:\Users\xxx\.matplotlib），
# 在沙箱或受限环境下可能不可写。这里把配置目录重定向到仓库内的 .mplconfig
# （可写、随仓库走），避免每次运行都报"无法保存字体缓存"的警告。
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))

# ---------------------------------------------------------------------------
# 1. 绘图后端：优先 matplotlib，否则回退到内置 SVG 绘制器
# ---------------------------------------------------------------------------
try:
    # "Agg" 是无窗口后端：只把图画成文件，不弹窗。
    # 好处：脚本可以在任何环境（包括服务器）跑，图直接存成 PNG。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True          # 标记：matplotlib 可用

    # 注册 Windows 自带的中文字体（微软雅黑等），避免图里中文变成方块。
    # 遍历几个常见字体文件，能用哪个用哪个。
    import matplotlib.font_manager as _fm
    for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttf",
               "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
        if os.path.exists(_f):
            try:
                _fm.fontManager.addfont(_f)                    # 注册该字体文件
                # 设置 matplotlib 默认无衬线字体为刚注册的字体（+ 备选 DejaVu）
                plt.rcParams["font.sans-serif"] = [_fm.FontProperties(fname=_f).get_name(),
                                                   "DejaVu Sans"]
                # 让坐标轴的负号用 ASCII "-" 而不是 Unicode 减号（避免显示成方块）
                plt.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
except Exception:
    # 如果没有 matplotlib：HAS_MPL=False，后面用 SVGPlot 画图
    HAS_MPL = False
    plt = None

# 画图用的配色：10 种颜色循环使用，让多条曲线容易区分
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def ensure_dir(path: str) -> str:
    """确保目录存在并返回该路径。

    用途：绘图输出统一放在各章的 _plots/ 目录，第一次保存图前先建目录。
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig, path: str) -> None:
    """把 matplotlib 图保存为 PNG 文件，并关闭图形对象释放内存。

    fig   : matplotlib 的 figure 对象（由 plt.subplots() 创建）
    path  : 保存路径，如 "chapters/ch01_xxx/_plots/fig1.png"
    """
    ensure_dir(os.path.dirname(path))          # 先确保目标目录存在
    fig.savefig(path, dpi=130, bbox_inches="tight")   # dpi=清晰度；bbox_inches 裁掉多余空白
    plt.close(fig)                             # 关图释放内存（否则连续画几十张图会占满内存）


# ---- 轻量 SVG 绘制器（matplotlib 不可用时的回退） -------------------------
class SVGPlot:
    """极简 SVG 画图器。

    SVG 是一种文本格式的图片（浏览器直接打开）。这个类只支持两类图形：
      1. 折线（line）：把一串 (x,y) 点连起来
      2. 散点（scatter）：画一堆圆点
    接口与 Figure 类保持一致，这样各章脚本不用关心后端是哪个。

    用法：
        p = SVGPlot("标题", "x轴", "y轴")
        p.line(x, y, label="曲线1")
        p.scatter(x, y, label="点")
        p.save("_plots/fig.png")   # 实际输出 .svg 文件
    """

    def __init__(self, title: str = "", xlabel: str = "", ylabel: str = ""):
        self.title = title       # 图标题
        self.xlabel = xlabel     # x 轴名字
        self.ylabel = ylabel     # y 轴名字
        self.series = []         # 所有要画的元素（线/点）的列表
        # 下面四个变量用来记录所有数据的取值范围，画坐标轴时要用
        self._xmin = self._xmax = self._ymin = self._ymax = None

    def _feed(self, xs, ys):
        """记录数据范围：把新数据的最大最小值并进已知范围。"""
        for v in xs:
            self._xmin = v if self._xmin is None else min(self._xmin, v)
            self._xmax = v if self._xmax is None else max(self._xmax, v)
        for v in ys:
            self._ymin = v if self._ymin is None else min(self._ymin, v)
            self._ymax = v if self._ymax is None else max(self._ymax, v)

    def line(self, x, y, label: str = ""):
        """画一条折线：x、y 是等长数组，逐点连成线。"""
        x = np.asarray(x, dtype=float)    # 转成 numpy 数组（如果传进来是列表）
        y = np.asarray(y, dtype=float)
        self._feed(x, y)                  # 记录范围
        self.series.append({"kind": "line", "x": x, "y": y, "label": label})

    def scatter(self, x, y, label: str = ""):
        """画散点：x、y 等长，每对 (x[i], y[i]) 画一个圆点。"""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self._feed(x, y)
        self.series.append({"kind": "scatter", "x": x, "y": y, "label": label})

    def save(self, path: str) -> str:
        """输出为 .svg 文件（扩展名自动改成 .svg）。

        这里的核心工作：把数据坐标换算成 SVG 画布上的像素坐标，
        然后生成 SVG 的 XML 文本写进文件。
        """
        path = os.path.splitext(path)[0] + ".svg"   # 把 .png 改成 .svg
        ensure_dir(os.path.dirname(path))

        # 画布大小：宽 820、高 560；四周留白（左 70 右 30 上 44 下 56）
        W, H, ML, MR, MT, MB = 820, 560, 70, 30, 44, 56
        if self._xmin is None:            # 空图兜底：给个默认范围
            self._xmin, self._xmax, self._ymin, self._ymax = 0, 1, 0, 1
        padx = max((self._xmax - self._xmin) * 0.06, 1e-9)   # x 方向留 6% 边距
        pady = max((self._ymax - self._ymin) * 0.08, 1e-9)   # y 方向留 8% 边距
        x0, x1 = self._xmin - padx, self._xmax + padx
        y0, y1 = self._ymin - pady, self._ymax + pady

        # 线性映射：数据坐标 -> 画布像素坐标（这就是"画图"的本质）
        def px(xv): return ML + (xv - x0) / (x1 - x0) * (W - ML - MR)
        def py(yv): return H - MB - (yv - y0) / (y1 - y0) * (H - MT - MB)

        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'font-family="Segoe UI, Microsoft YaHei, sans-serif">']
        # 网格线与坐标刻度（把范围均匀分成 5 段）
        for i in range(6):
            fx = x0 + (x1 - x0) * i / 5     # 第 i 条网格线对应的数据值
            fy = y0 + (y1 - y0) * i / 5
            gx, gy = px(fx), py(fy)
            parts.append(f'<line x1="{gx:.1f}" y1="{MB:.0f}" x2="{gx:.1f}" y2="{H-MT:.0f}" '
                         f'stroke="#e8e8e8" stroke-width="1"/>')
            parts.append(f'<line x1="{ML:.0f}" y1="{gy:.1f}" x2="{W-MR:.0f}" y2="{gy:.1f}" '
                         f'stroke="#e8e8e8" stroke-width="1"/>')
            # 刻度文字
            parts.append(f'<text x="{gx:.1f}" y="{H-MB+18:.0f}" font-size="12" '
                         f'text-anchor="middle" fill="#444">{fx:.2g}</text>')
            parts.append(f'<text x="{ML-8:.0f}" y="{gy+4:.1f}" font-size="12" '
                         f'text-anchor="end" fill="#444">{fy:.2g}</text>')
        # 坐标轴主线
        parts.append(f'<line x1="{ML:.0f}" y1="{H-MB:.0f}" x2="{W-MR:.0f}" y2="{H-MB:.0f}" '
                     f'stroke="#333" stroke-width="1.5"/>')
        parts.append(f'<line x1="{ML:.0f}" y1="{MB:.0f}" x2="{ML:.0f}" y2="{H-MB:.0f}" '
                     f'stroke="#333" stroke-width="1.5"/>')
        parts.append(f'<text x="{(W)/2:.0f}" y="26" font-size="16" text-anchor="middle" '
                     f'fill="#111">{self.title}</text>')
        parts.append(f'<text x="{(W)/2:.0f}" y="{H-10:.0f}" font-size="13" '
                     f'text-anchor="middle" fill="#111">{self.xlabel}</text>')

        # 依次绘制每条曲线/每个散点
        for i, s in enumerate(self.series):
            c = _PALETTE[i % len(_PALETTE)]            # 循环取颜色
            if s["kind"] == "line":
                # 把所有数据点映射到像素坐标，拼成 "x1,y1 x2,y2 ..." 的字符串
                pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(s["x"], s["y"]))
                parts.append(f'<polyline points="{pts}" fill="none" stroke="{c}" '
                             f'stroke-width="2.2"/>')
            else:
                # 每个点画一个圆
                for x, y in zip(s["x"], s["y"]):
                    parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4" '
                                 f'fill="{c}" opacity="0.85"/>')
            if s["label"]:                             # 右上角画图例
                lx, ly = W - MR - 110, MB + 18 + i * 20
                parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" '
                             f'stroke="{c}" stroke-width="2.2"/>')
                parts.append(f'<text x="{lx+30}" y="{ly+4}" font-size="12" fill="#333">{s["label"]}</text>')
        parts.append("</svg>")

        # 把 XML 文本写入文件（utf-8 编码，中文正常显示）
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        return path


class Figure:
    """统一绘图入口：有 matplotlib 用真图，没有则用 SVGPlot。

    各章脚本的用法（不需要关心内部）：
        fig = Figure("标题", "x轴", "y轴")
        fig.line(x, y, label="曲线")
        fig.scatter(x, y, label="点")
        fig.save("_plots/fig.png")
    """

    def __init__(self, title: str = "", xlabel: str = "", ylabel: str = ""):
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self._series = []                              # 记录画过的所有元素
        if HAS_MPL:
            self._fig, self._ax = plt.subplots(figsize=(7.2, 4.8))  # 创建画布

    def line(self, x, y, label: str = ""):
        """画线：转发给 matplotlib 或 SVGPlot。"""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if HAS_MPL:
            self._ax.plot(x, y, lw=2.0, label=label)   # lw=线宽
        else:
            self._series.append(("line", x, y, label))

    def scatter(self, x, y, label: str = ""):
        """画散点。"""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if HAS_MPL:
            self._ax.scatter(x, y, s=26, label=label, zorder=3)   # zorder=点画在线上层
        else:
            self._series.append(("scatter", x, y, label))

    def save(self, path: str) -> str:
        """保存图片：matplotlib -> PNG；否则 -> SVG。"""
        if HAS_MPL:
            self._ax.set_title(self.title)             # 设置标题/轴名
            self._ax.set_xlabel(self.xlabel)
            self._ax.set_ylabel(self.ylabel)
            self._ax.grid(alpha=0.3)                   # 浅色网格，帮助读图
            handles, labels = self._ax.get_legend_handles_labels()
            if labels:                                 # 有标签才画图例
                self._ax.legend(fontsize=9)
            save_figure(self._fig, path)
            return path
        # 无 matplotlib：把记录的元素转发给 SVGPlot 画
        sp = SVGPlot(self.title, self.xlabel, self.ylabel)
        for kind, x, y, label in self._series:
            if kind == "line":
                sp.line(x, y, label)
            else:
                sp.scatter(x, y, label)
        return sp.save(path)


# ---------------------------------------------------------------------------
# 2. 误差度量
# ---------------------------------------------------------------------------
def rms_error(t_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差 E_RMS（书中 1.2 节）。

    定义：E_RMS = sqrt( mean( (y_pred - t_true)^2 ) )
    即"平均平方误差再开根号"。为什么开根号？让误差量纲和原始数据一致。

    用途：Ch1 里衡量多项式拟合"差多少"，训练/测试误差各算一个。
    """
    t_true = np.asarray(t_true, dtype=float)   # 真实值（目标）
    y_pred = np.asarray(y_pred, dtype=float)   # 预测值
    # (y_pred - t_true)**2 : 逐元素相减再平方（向量化，等价于循环）
    # .mean() : 对所有元素取平均
    # np.sqrt : 开根号
    return float(np.sqrt(np.mean((y_pred - t_true) ** 2)))


# ---------------------------------------------------------------------------
# 3. 多项式基函数与最小二乘（第 1 章 / 第 4 章共用）
# ---------------------------------------------------------------------------
def poly_design_matrix(x: np.ndarray, M: int) -> np.ndarray:
    """多项式基函数设计矩阵 Phi，形状 (N, M+1)。

    含义：把"输入 x"变成"特征向量"。
      Phi[n, j] = x_n^j   （j = 0..M，即 x 的 0 次方到 M 次方）

    为什么需要它？模型 y(x, w) = Σ_{j=0}^{M} w_j x^j 可以写成矩阵形式
      y = Phi @ w
    这样"拟合参数 w"就变成了解线性方程组，后面用闭式解一步算完。

    例：x = [0.5]，M=2 -> Phi = [[1, 0.5, 0.25]]
    """
    x = np.asarray(x, dtype=float)
    # np.vander 生成范德蒙德矩阵：
    #   vander([0.5], 3, increasing=True) = [[1, 0.5, 0.25]]
    # increasing=True 表示指数从小到大（j=0,1,2）
    return np.vander(x, M + 1, increasing=True)


def poly_fit_least_squares(x: np.ndarray, t: np.ndarray, M: int,
                           lam: float = 0.0, method: str = "lstsq") -> np.ndarray:
    """拟合 M 阶多项式（可选 L2 正则），返回系数 w（长度 M+1）。

    数学上的闭式解（书中 1.2 节）：
        w* = (Phi^T Phi + lam*I)^{-1} Phi^T t
    即：让误差 E(w) = ½Σ(y_n - t_n)² + (λ/2)‖w‖² 最小的 w。

    但注意：M 较大时 Phi^T Phi 病态（条件数极大），直接求逆会数值崩溃
    （Ch1 里实测 M=9 时偏差达 653！）。所以默认用基于 SVD 的 lstsq
    （数值上最稳定）。method="solve" 才走正规方程，用于验证公式本身
    （小 M 时两者一致）。

    参数：
        x, t : 训练数据（输入、输出）
        M    : 多项式阶数（模型容量）
        lam  : L2 正则化强度（0 = 不正则化）
        method : "lstsq"（默认，稳定）或 "solve"（正规方程）
    """
    Phi = poly_design_matrix(x, M)          # 设计矩阵 (N, M+1)
    t = np.asarray(t, dtype=float)
    if method == "solve":
        # 正规方程：(PhiᵀPhi + λI) w = Phiᵀt，用 np.linalg.solve 解
        A = Phi.T @ Phi + lam * np.eye(M + 1)   # (M+1, M+1)
        return np.linalg.solve(A, Phi.T @ t)
    if lam == 0.0:
        # 无正则：直接最小二乘（SVD 方法，稳定）
        w, *_ = np.linalg.lstsq(Phi, t, rcond=None)
        return w
    # 有正则：仍然走正规方程（λ 改善了条件数，够稳定）
    A = Phi.T @ Phi + lam * np.eye(M + 1)
    return np.linalg.solve(A, Phi.T @ t)


def poly_eval(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """用系数 w 计算多项式在 x 处的值：y(x, w) = Σ_j w_j x^j。

    实现：y = Phi(x) @ w —— 先构造设计矩阵再乘系数，一次算完所有 x。
    """
    x = np.asarray(x, dtype=float)
    return poly_design_matrix(x, len(w) - 1) @ w


# ---------------------------------------------------------------------------
# 4. 数值梯度检验（书中 8.5 节数值微分思想的直接应用）
# ---------------------------------------------------------------------------
def check_gradient(f, grad, x0, eps: float = 1e-6, verbose: bool = True):
    """用中心有限差分验证解析梯度 grad 是否与函数 f 一致。

    为什么需要它？
      手推的反向传播公式（解析梯度）可能写错。数值微分用定义
        f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
      直接算梯度（不需要推导，就是定义），所以可以用来验证解析梯度对不对。
      每章脚本都用它做"自检"。

    参数
    ----
    f    : 标量函数 f(x) -> float（x 是一维 numpy 数组）
    grad : 梯度函数 grad(x) -> 与 x 同形状的数组
    x0   : 检验点（一维 numpy 数组）
    eps  : 差分步长（h）
    verbose : 是否打印结果

    返回 (max_abs_diff, 是否通过)。
    """
    x0 = np.asarray(x0, dtype=float)
    g_analytic = np.asarray(grad(x0), dtype=float)   # 解析梯度（要验证的对象）
    g_numeric = np.zeros_like(x0)                    # 数值梯度（参照标准）
    for i in range(x0.size):                         # 对每个维度单独做差分
        xp = x0.copy(); xp[i] += eps                 # 第 i 维 + h
        xm = x0.copy(); xm[i] -= eps                 # 第 i 维 - h
        # 中心差分公式：(f(x+h) - f(x-h)) / 2h
        g_numeric[i] = (f(xp) - f(xm)) / (2 * eps)
    diff = np.abs(g_analytic - g_numeric)            # 逐维比较
    max_diff = float(diff.max())
    ok = max_diff < 1e-4 * max(1.0, float(np.abs(g_analytic).max()))
    if verbose:
        print(f"  数值梯度检验: 最大绝对误差 = {max_diff:.3e} -> {'通过 ✓' if ok else '失败 ✗'}")
    return max_diff, ok


# ---------------------------------------------------------------------------
# 5. 概率与信息论工具（第 2-3 章共用）
# ---------------------------------------------------------------------------
def gaussian_pdf(x, mu: float = 0.0, sigma2: float = 1.0) -> np.ndarray:
    """一维高斯概率密度（书中 2.3 节）。

    p(x) = 1/sqrt(2πσ²) * exp( -(x-mu)² / (2σ²) )

    逐行解释：
      (x - mu)**2       : 每个 x 与均值距离的平方
      / (2 * sigma2)    : 除以 2 倍方差（控制"尖/扁"）
      np.exp(...)       : e 的幂次
      / np.sqrt(2πσ²)   : 归一化常数，保证 ∫p(x)dx = 1

    x 可以是数组，返回同样形状的数组（向量化）。
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * (x - mu) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)


def multivariate_gaussian_pdf(x, mu, Sigma) -> float:
    """多维高斯概率密度（书中 2.4 节）。

    p(x) = (2π)^(-D/2) |Σ|^(-1/2) exp( -½ (x-mu)ᵀ Σ⁻¹ (x-mu) )

    与一维的区别：
      - 标量方差 σ² 换成协方差矩阵 Sigma（描述各维方差 + 维间相关）
      - (x-mu)²/(2σ²) 换成二次型 (x-mu)ᵀΣ⁻¹(x-mu)/2（马氏距离）
      - 归一化常数用 det(Σ)

    实现上：用 Cholesky 分解和 solve 求 Σ⁻¹ 与 log|Σ|，数值更稳定。
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    D = mu.size                                       # 维度
    diff = x - mu
    L = np.linalg.cholesky(Sigma)                     # Sigma = L Lᵀ
    maha = float(diff @ np.linalg.solve(Sigma, diff)) # 马氏距离平方
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))  # log|Σ| = 2Σlog(L 对角)
    return float(np.exp(-0.5 * (maha + D * np.log(2 * np.pi) + logdet)))


def entropy(p: np.ndarray, unit: str = "nats") -> float:
    """离散熵 H(p) = -Σ p_i ln p_i（书中 2.5.1 节）。

    直观理解：H 度量"不确定性"。硬币两面均匀（p=0.5,0.5）时 H=1 bit，
    完全确定（p=1,0）时 H=0。越均匀，越难预测，熵越大。

    输入必须是合法概率分布（非负、和为 1）。
    """
    p = np.asarray(p, dtype=float)
    assert np.all(p >= 0) and abs(p.sum() - 1.0) < 1e-9, "p 必须是合法概率分布"
    p = p[p > 0]                                      # 0 ln 0 约定为 0，剔除
    base = 2.0 if unit == "bits" else np.e            # bits 用 log2，nats 用 ln
    return float(-np.sum(p * np.log(p)) / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL 散度 KL(p||q) = Σ p_i ln(p_i / q_i)（书中 2.5.5 节）。

    度量"用 q 近似 p 的损失"。性质：
      - KL >= 0，等号当且仅当 p == q
      - 不对称：KL(p||q) != KL(q||p)（所以叫"散度"不是"距离"）

    注意：这里要求 p、q 严格为正（0 会出问题），代码里用 assert 强制。
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    assert p.shape == q.shape and np.all(p > 0) and np.all(q > 0), "需要严格正的概率向量"
    # p * (ln p - ln q) 等价于 p * ln(p/q)（对数的差 = 商的对数）
    return float(np.sum(p * (np.log(p) - np.log(q))))


def mutual_information(joint: np.ndarray) -> float:
    """互信息 I(X;Y)（书中 2.5.7 节），输入联合分布矩阵 joint[i,j] = p(X=i, Y=j)。

    I(X;Y) = ΣΣ p(x,y) ln( p(x,y) / (p(x)p(y)) ) = H(X) + H(Y) - H(X,Y)

    直观理解：观测 Y 之后，X 的不确定性减少了多少。独立时 I=0。
    """
    joint = np.asarray(joint, dtype=float)
    assert joint.ndim == 2 and np.all(joint > 0) and abs(joint.sum() - 1.0) < 1e-9
    px = joint.sum(axis=1)        # 对列求和 -> p(X)（边际化，和规则）
    py = joint.sum(axis=0)        # 对行求和 -> p(Y)
    outer = np.outer(px, py)      # p(X)p(Y)：独立时的联合（外积）
    # joint / outer : p(x,y) / (p(x)p(y))，取 ln，再乘 p(x,y) 求和
    return float(np.sum(joint * (np.log(joint) - np.log(outer))))


# ---- 第 3 章的离散分布 ----
def bernoulli_pmf(x, mu: float) -> float:
    """伯努利分布 p(x|mu) = mu^x (1-mu)^(1-x)，x ∈ {0,1}（书中 3.1 节）。

    含义：抛一次硬币（正面概率 mu）。x=1 时 p=mu；x=0 时 p=1-mu。
    """
    assert 0 <= mu <= 1
    return float((mu ** x) * ((1.0 - mu) ** (1 - x)))


def binomial_pmf(k, n: int, mu: float) -> float:
    """二项分布 Bin(k|n,mu) = C(n,k) mu^k (1-mu)^(n-k)（书中 3.1 节）。

    含义：抛 n 次硬币（正面概率 mu），恰好 k 次正面的概率。
    C(n,k) 是从 n 个里选 k 个的组合数（math.comb 计算）。
    """
    from math import comb
    return float(comb(n, k) * (mu ** k) * ((1.0 - mu) ** (n - k)))


def multinomial_pmf(counts, probs) -> float:
    """多项分布 Mult(n1..nK | mu1..muK)（书中 3.1 节）。

    伯努利/二项的推广：一次实验有 K 种结果（概率 probs），
    做 n 次实验后各结果出现 counts 次的概率。
    """
    import math
    counts = np.asarray(counts, dtype=int)
    probs = np.asarray(probs, dtype=float)
    assert counts.ndim == 1 and counts.size == probs.size
    assert np.all(counts >= 0) and abs(probs.sum() - 1.0) < 1e-9
    n = int(counts.sum())                     # 总次数
    coef = math.factorial(n)                  # n! / Π n_k!：组合数
    for c in counts:
        coef //= math.factorial(int(c))
    return float(coef * np.prod(probs ** counts))   # × Π mu_k^{n_k}


# ---------------------------------------------------------------------------
# 6. 第 1 章的合成数据
# ---------------------------------------------------------------------------
def gen_sin_data(N: int = 10, sigma: float = 0.3, seed: int = 42):
    """生成书中 1.2 节的回归数据：t = sin(2πx) + N(0, sigma^2)。

    x 在 [0, 1] 上均匀取 N 个点；t 是在真实函数 sin(2πx) 上加高斯噪声。

    为什么要固定 seed？
      随机数生成器需要种子，同一个 seed 每次生成完全相同的数据。
      这样书中/代码里的一切数值结果都可以复现，不会每次跑不一样。

    返回 (x, t)。
    """
    rng = np.random.default_rng(seed)   # 创建一个带种子的随机数生成器
    x = np.linspace(0.0, 1.0, N)        # [0,1] 上均匀取 N 个点
    # sin(2πx) 是真实函数；rng.normal(0, sigma, N) 生成 N 个高斯噪声
    t = np.sin(2 * np.pi * x) + rng.normal(0.0, sigma, size=N)
    return x, t
