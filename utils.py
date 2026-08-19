# -*- coding: utf-8 -*-
"""
dlfc-code 共享工具模块
======================
配套《Deep Learning: Foundations and Concepts》(Bishop & Bishop, Springer 2023)
逐章代码化的公共工具：数据生成、多项式基、误差度量、数值梯度检验、绘图。

绘图策略：
    优先使用 matplotlib（若已安装）；否则回退到内置的轻量 SVG 绘制器，
    保证每个章节脚本在没有 matplotlib 的机器上也能运行并产出图片。
"""
from __future__ import annotations

import os
import sys

# 仓库内 .pylib 优先：这里放着"手动解压安装"的第三方包（如 matplotlib）。
# 这样即使系统 site-packages 装不了包，仓库本身也自包含、可移植。
_PYLIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pylib")
if os.path.isdir(_PYLIB) and _PYLIB not in sys.path:
    sys.path.insert(0, _PYLIB)

import numpy as np

# matplotlib 的字体/缓存默认写到用户目录（C:\Users\eric\.matplotlib），
# 在沙箱或受限环境下可能不可写。这里重定向到仓库内的 .mplconfig（可写、可移植）。
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))

# ---------------------------------------------------------------------------
# 1. 绘图后端：matplotlib 优先，SVG 回退
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # 无窗口后端：只保存图片文件，不弹窗
    import matplotlib.pyplot as plt

    HAS_MPL = True

    # 注册 Windows 自带的中文字体，避免图标题/图例出现方块
    import matplotlib.font_manager as _fm
    for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttf",
               "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
        if os.path.exists(_f):
            try:
                _fm.fontManager.addfont(_f)
                plt.rcParams["font.sans-serif"] = [_fm.FontProperties(fname=_f).get_name(),
                                                   "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
except Exception:  # pragma: no cover - 无 matplotlib 环境
    HAS_MPL = False
    plt = None

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def ensure_dir(path: str) -> str:
    """确保目录存在并返回该路径（绘图输出统一放在各章 _plots/ 下）。"""
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig, path: str) -> None:
    """保存 matplotlib 图（png 优先）。"""
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---- 轻量 SVG 绘制器（matplotlib 不可用时的回退） -------------------------
class SVGPlot:
    """极简 SVG 折线/散点绘图器，接口与常用绘图习惯对齐。

    用法：
        p = SVGPlot("标题", "x轴", "y轴")
        p.line(x, y, label="M=9")
        p.scatter(x, y, label="训练数据")
        p.save("_plots/fig.png")   # 实际写出 .svg
    """

    def __init__(self, title: str = "", xlabel: str = "", ylabel: str = ""):
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.series: list[dict] = []
        self._xmin = self._xmax = self._ymin = self._ymax = None

    def _feed(self, xs, ys):
        for v in xs:
            self._xmin = v if self._xmin is None else min(self._xmin, v)
            self._xmax = v if self._xmax is None else max(self._xmax, v)
        for v in ys:
            self._ymin = v if self._ymin is None else min(self._ymin, v)
            self._ymax = v if self._ymax is None else max(self._ymax, v)

    def line(self, x, y, label: str = ""):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self._feed(x, y)
        self.series.append({"kind": "line", "x": x, "y": y, "label": label})

    def scatter(self, x, y, label: str = ""):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self._feed(x, y)
        self.series.append({"kind": "scatter", "x": x, "y": y, "label": label})

    def save(self, path: str) -> str:
        """输出为 .svg 文件（扩展名自动改为 .svg）。"""
        path = os.path.splitext(path)[0] + ".svg"
        ensure_dir(os.path.dirname(path))

        W, H, ML, MR, MT, MB = 820, 560, 70, 30, 44, 56
        if self._xmin is None:  # 空图兜底
            self._xmin, self._xmax, self._ymin, self._ymax = 0, 1, 0, 1
        padx = max((self._xmax - self._xmin) * 0.06, 1e-9)
        pady = max((self._ymax - self._ymin) * 0.08, 1e-9)
        x0, x1 = self._xmin - padx, self._xmax + padx
        y0, y1 = self._ymin - pady, self._ymax + pady

        def px(xv): return ML + (xv - x0) / (x1 - x0) * (W - ML - MR)
        def py(yv): return H - MB - (yv - y0) / (y1 - y0) * (H - MT - MB)

        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'font-family="Segoe UI, Microsoft YaHei, sans-serif">']
        # 网格与坐标轴
        for i in range(6):
            fx = x0 + (x1 - x0) * i / 5
            fy = y0 + (y1 - y0) * i / 5
            gx, gy = px(fx), py(fy)
            parts.append(f'<line x1="{gx:.1f}" y1="{MB:.0f}" x2="{gx:.1f}" y2="{H-MT:.0f}" '
                         f'stroke="#e8e8e8" stroke-width="1"/>')
            parts.append(f'<line x1="{ML:.0f}" y1="{gy:.1f}" x2="{W-MR:.0f}" y2="{gy:.1f}" '
                         f'stroke="#e8e8e8" stroke-width="1"/>')
            parts.append(f'<text x="{gx:.1f}" y="{H-MB+18:.0f}" font-size="12" '
                         f'text-anchor="middle" fill="#444">{fx:.2g}</text>')
            parts.append(f'<text x="{ML-8:.0f}" y="{gy+4:.1f}" font-size="12" '
                         f'text-anchor="end" fill="#444">{fy:.2g}</text>')
        parts.append(f'<line x1="{ML:.0f}" y1="{H-MB:.0f}" x2="{W-MR:.0f}" y2="{H-MB:.0f}" '
                     f'stroke="#333" stroke-width="1.5"/>')
        parts.append(f'<line x1="{ML:.0f}" y1="{MB:.0f}" x2="{ML:.0f}" y2="{H-MB:.0f}" '
                     f'stroke="#333" stroke-width="1.5"/>')
        parts.append(f'<text x="{(W)/2:.0f}" y="26" font-size="16" text-anchor="middle" '
                     f'fill="#111">{self.title}</text>')
        parts.append(f'<text x="{(W)/2:.0f}" y="{H-10:.0f}" font-size="13" '
                     f'text-anchor="middle" fill="#111">{self.xlabel}</text>')

        for i, s in enumerate(self.series):
            c = _PALETTE[i % len(_PALETTE)]
            if s["kind"] == "line":
                pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(s["x"], s["y"]))
                parts.append(f'<polyline points="{pts}" fill="none" stroke="{c}" '
                             f'stroke-width="2.2"/>')
            else:
                for x, y in zip(s["x"], s["y"]):
                    parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4" '
                                 f'fill="{c}" opacity="0.85"/>')
            if s["label"]:
                lx, ly = W - MR - 110, MB + 18 + i * 20
                parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" '
                             f'stroke="{c}" stroke-width="2.2"/>')
                parts.append(f'<text x="{lx+30}" y="{ly+4}" font-size="12" fill="#333">{s["label"]}</text>')
        parts.append("</svg>")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        return path


class Figure:
    """统一绘图入口：有 matplotlib 用真图，没有则用 SVGPlot。"""

    def __init__(self, title: str = "", xlabel: str = "", ylabel: str = ""):
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self._series: list[tuple[str, np.ndarray, np.ndarray, str]] = []
        if HAS_MPL:
            self._fig, self._ax = plt.subplots(figsize=(7.2, 4.8))

    def line(self, x, y, label: str = ""):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if HAS_MPL:
            self._ax.plot(x, y, lw=2.0, label=label)
        else:
            self._series.append(("line", x, y, label))

    def scatter(self, x, y, label: str = ""):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if HAS_MPL:
            self._ax.scatter(x, y, s=26, label=label, zorder=3)
        else:
            self._series.append(("scatter", x, y, label))

    def save(self, path: str) -> str:
        if HAS_MPL:
            self._ax.set_title(self.title)
            self._ax.set_xlabel(self.xlabel)
            self._ax.set_ylabel(self.ylabel)
            self._ax.grid(alpha=0.3)
            handles, labels = self._ax.get_legend_handles_labels()
            if labels:
                self._ax.legend(fontsize=9)
            save_figure(self._fig, path)
            return path
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

    定义：E_RMS = sqrt( 2 * E(w*) / N )，其中 E 是平方和误差。
    即 E_RMS = sqrt( mean( (y_pred - t_true)^2 ) )。
    """
    t_true = np.asarray(t_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - t_true) ** 2)))


# ---------------------------------------------------------------------------
# 6. 概率与信息论工具（第 2-3 章共用）
# ---------------------------------------------------------------------------
def gaussian_pdf(x, mu: float = 0.0, sigma2: float = 1.0) -> np.ndarray:
    """一维高斯概率密度（书中 2.3 节）。

    p(x) = 1/sqrt(2πσ²) * exp( -(x-mu)² / (2σ²) )
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * (x - mu) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)


def multivariate_gaussian_pdf(x, mu, Sigma) -> float:
    """多维高斯概率密度（书中 2.4 节）。

    p(x) = (2π)^(-D/2) |Σ|^(-1/2) exp( -½ (x-mu)ᵀ Σ⁻¹ (x-mu) )
    用 cholesky/solve 保证数值稳定。
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    D = mu.size
    diff = x - mu
    L = np.linalg.cholesky(Sigma)
    maha = float(diff @ np.linalg.solve(Sigma, diff))
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return float(np.exp(-0.5 * (maha + D * np.log(2 * np.pi) + logdet)))


def entropy(p: np.ndarray, unit: str = "nats") -> float:
    """离散熵 H(p) = -Σ p_i ln p_i（书中 2.5.1 节）。

    输入必须是合法概率分布（非负、和为 1）。unit='bits' 时以 2 为底。
    """
    p = np.asarray(p, dtype=float)
    assert np.all(p >= 0) and abs(p.sum() - 1.0) < 1e-9, "p 必须是合法概率分布"
    p = p[p > 0]  # 0 ln 0 = 0
    base = 2.0 if unit == "bits" else np.e
    return float(-np.sum(p * np.log(p)) / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL 散度 KL(p||q) = Σ p_i ln(p_i / q_i)（书中 2.5.5 节）。

    性质：KL >= 0，等号当且仅当 p == q；不对称（KL(p||q) != KL(q||p)）。
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    assert p.shape == q.shape and np.all(p > 0) and np.all(q > 0), "需要严格正的概率向量"
    return float(np.sum(p * (np.log(p) - np.log(q))))


def mutual_information(joint: np.ndarray) -> float:
    """互信息 I(X;Y)（书中 2.5.7 节），输入联合分布矩阵 joint[i,j] = p(X=i, Y=j)。

    I(X;Y) = ΣΣ p(x,y) ln( p(x,y) / (p(x)p(y)) ) = H(X) + H(Y) - H(X,Y)
    """
    joint = np.asarray(joint, dtype=float)
    assert joint.ndim == 2 and np.all(joint > 0) and abs(joint.sum() - 1.0) < 1e-9
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    outer = np.outer(px, py)
    return float(np.sum(joint * (np.log(joint) - np.log(outer))))


# ---- 第 3 章的离散分布 ----
def bernoulli_pmf(x, mu: float) -> float:
    """伯努利分布 p(x|mu) = mu^x (1-mu)^(1-x)，x ∈ {0,1}（书中 3.1 节）。"""
    assert 0 <= mu <= 1
    return float((mu ** x) * ((1.0 - mu) ** (1 - x)))


def binomial_pmf(k, n: int, mu: float) -> float:
    """二项分布 Bin(k|n,mu) = C(n,k) mu^k (1-mu)^(n-k)（书中 3.1 节）。"""
    from math import comb
    return float(comb(n, k) * (mu ** k) * ((1.0 - mu) ** (n - k)))


def multinomial_pmf(counts, probs) -> float:
    """多项分布 Mult(n1..nK | mu1..muK)（书中 3.1 节）。

    counts/probs 为等长数组；阶乘用 math.factorial 计算。
    """
    import math
    counts = np.asarray(counts, dtype=int)
    probs = np.asarray(probs, dtype=float)
    assert counts.ndim == 1 and counts.size == probs.size
    assert np.all(counts >= 0) and abs(probs.sum() - 1.0) < 1e-9
    n = int(counts.sum())
    coef = math.factorial(n)
    for c in counts:
        coef //= math.factorial(int(c))
    return float(coef * np.prod(probs ** counts))



# ---------------------------------------------------------------------------
# 3. 多项式基函数与最小二乘（第 1 章 / 第 4 章共用）
# ---------------------------------------------------------------------------
def poly_design_matrix(x: np.ndarray, M: int) -> np.ndarray:
    """多项式基函数设计矩阵 Phi，形状 (N, M+1)。

    Phi[n, j] = x_n^j，j = 0..M。对应书中 y(x, w) = Σ_{j=0}^{M} w_j x^j。
    """
    x = np.asarray(x, dtype=float)
    return np.vander(x, M + 1, increasing=True)


def poly_fit_least_squares(x: np.ndarray, t: np.ndarray, M: int,
                           lam: float = 0.0, method: str = "lstsq") -> np.ndarray:
    """拟合 M 阶多项式（可选 L2 正则），返回系数 w。

    数学上的闭式解（书中 1.2 节）：
        w* = (Phi^T Phi + lam*I)^{-1} Phi^T t
    但 M 较大时 Phi^T Phi 病态（条件数极大），直接解会数值崩溃，
    所以默认用基于 SVD 的 lstsq（更稳定）。method="solve" 才走正规方程，
    用于验证公式本身（小 M 时两者一致）。
    """
    Phi = poly_design_matrix(x, M)
    t = np.asarray(t, dtype=float)
    if method == "solve":
        A = Phi.T @ Phi + lam * np.eye(M + 1)
        return np.linalg.solve(A, Phi.T @ t)
    if lam == 0.0:
        w, *_ = np.linalg.lstsq(Phi, t, rcond=None)
        return w
    A = Phi.T @ Phi + lam * np.eye(M + 1)
    return np.linalg.solve(A, Phi.T @ t)


def poly_eval(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """对系数 w 求多项式在 x 处的值：y(x, w) = Σ_j w_j x^j。"""
    x = np.asarray(x, dtype=float)
    return poly_design_matrix(x, len(w) - 1) @ w


# ---------------------------------------------------------------------------
# 4. 数值梯度检验（书中 8.5 节数值微分思想的直接应用）
# ---------------------------------------------------------------------------
def check_gradient(f, grad, x0, eps: float = 1e-6, verbose: bool = True):
    """用中心有限差分验证解析梯度 grad 是否与函数 f 一致。

    参数
    ----
    f    : 标量函数 f(x) -> float（x 为一维 numpy 数组）
    grad : 梯度函数 grad(x) -> 与 x 同形状的数组
    x0   : 检验点（一维 numpy 数组）
    eps  : 差分步长

    返回 (max_abs_diff, 是否通过)。书中 8.5 节：数值微分是验证解析梯度的标准工具。
    """
    x0 = np.asarray(x0, dtype=float)
    g_analytic = np.asarray(grad(x0), dtype=float)
    g_numeric = np.zeros_like(x0)
    for i in range(x0.size):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        g_numeric[i] = (f(xp) - f(xm)) / (2 * eps)
    diff = np.abs(g_analytic - g_numeric)
    max_diff = float(diff.max())
    ok = max_diff < 1e-4 * max(1.0, float(np.abs(g_analytic).max()))
    if verbose:
        print(f"  数值梯度检验: 最大绝对误差 = {max_diff:.3e} -> {'通过 ✓' if ok else '失败 ✗'}")
    return max_diff, ok


# ---------------------------------------------------------------------------
# 5. 第 1 章的合成数据
# ---------------------------------------------------------------------------
def gen_sin_data(N: int = 10, sigma: float = 0.3, seed: int = 42):
    """生成书中 1.2 节的回归数据：t = sin(2πx) + N(0, sigma^2)。

    x 在 [0, 1] 上均匀取 N 个点，噪声标准差 sigma=0.3（与书一致）。
    返回 (x, t)。
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, N)
    t = np.sin(2 * np.pi * x) + rng.normal(0.0, sigma, size=N)
    return x, t
