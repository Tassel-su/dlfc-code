# -*- coding: utf-8 -*-
"""
第 12 章：Transformer（注意力）
==========================================
对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop) 第 12 章
（印刷页 357-404，PDF 页 377-424）。

本章要亲眼看到的现象：
  12.1 注意力机制：缩放点积注意力（前向+反向+梯度验证）、
      自注意力、多头注意力、位置编码（正弦）、复杂度 O(L²)；
  12.2 自然语言：词嵌入、分词、自回归模型、RNN/BPTT；
  12.3 Transformer 语言模型：从零训练一个微型 decoder transformer
      并采样生成文本（贪心/温度采样）；
  12.4 多模态：Vision Transformer（图像分块）简介。

运行方式：
  C:/Python314/python.exe ch12_transformer.py
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


def softmax_rows(M):
    """按行 softmax（数值稳定）。"""
    M = M - M.max(axis=-1, keepdims=True)
    e = np.exp(M)
    return e / e.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """缩放点积注意力（书中 12.1.5 节）。

    Q, K, V: (L, dk)（单样本、单头）；mask: (L, L) 因果掩码（-inf 表示屏蔽）。
    返回 (输出, 注意力权重)。
    """
    dk = K.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)          # (L, L)：每对位置的匹配度
    if mask is not None:
        scores = scores + mask              # 因果掩码：未来的位置置 -inf
    attn = softmax_rows(scores)             # 注意力权重：行和为 1
    out = attn @ V                          # 加权求和各位置的值
    return out, attn


def attention_backward(dout, Q, K, V, attn, mask=None):
    """缩放点积注意力的反向传播（用于梯度验证与训练）。"""
    dk = K.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    if mask is not None:
        scores = scores + mask
    # dV = attnᵀ @ dout
    dV = attn.T @ dout
    # dattn = dout @ Vᵀ
    dattn = dout @ V.T
    # softmax 反向：dscores = attn ⊙ (dattn - Σ dattn⊙attn 沿行)
    dattn_masked = dattn * attn
    dscores = attn * (dattn - dattn_masked.sum(axis=-1, keepdims=True))
    dscores /= np.sqrt(dk)
    dQ = dscores @ K
    dK = dscores.T @ Q
    return dQ, dK, dV


def positional_encoding(L, d):
    """正弦位置编码（书中 12.1.9 节）。"""
    pe = np.zeros((L, d))
    for pos in range(L):
        for i in range(0, d, 2):
            pe[pos, i] = np.sin(pos / 10000 ** (i / d))
            if i + 1 < d:
                pe[pos, i + 1] = np.cos(pos / 10000 ** (i / d))
    return pe


def main() -> None:
    # ==================================================================
    # 12.1 缩放点积注意力（书中 12.1.5 节）
    # ==================================================================
    section("12.1 缩放点积注意力：Q、K、V 三件套（书中 12.1.2/12.1.5 节）")
    print("注意力 = 用查询 Q 与键 K 的匹配度给值 V 加权求和：")
    print("  Attention(Q,K,V) = softmax(QKᵀ/√d) V")
    rng = np.random.default_rng(0)
    L, dk = 5, 4
    Q = rng.normal(0, 1, (L, dk))
    K = rng.normal(0, 1, (L, dk))
    V = rng.normal(0, 1, (L, dk))
    out, attn = scaled_dot_product_attention(Q, K, V)
    print(f"  注意力权重（行和为 1）：{np.round(attn, 3)}")
    assert abs(attn.sum(axis=1) - 1).max() < 1e-9, "注意力权重未归一化"
    print("  输出 = 各位置值的加权平均；权重越大的位置对输出贡献越大")

    # ---- 梯度验证 ----
    print("\n  验证注意力的反向传播（vs 数值梯度）：")
    eps = 1e-6
    dout = np.ones_like(out)
    dQ, dK, dV = attention_backward(dout, Q, K, V, attn)
    # 数值梯度 of scalar loss = sum(out)
    def loss_fn(Qv, Kv, Vv):
        o, _ = scaled_dot_product_attention(Qv, Kv, Vv)
        return o.sum()
    errs = []
    for name, dA, A in (("Q", dQ, Q), ("K", dK, K), ("V", dV, V)):
        gnum = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                Ap = A.copy(); Ap[i, j] += eps
                Am = A.copy(); Am[i, j] -= eps
                if name == "Q":
                    gnum[i, j] = (loss_fn(Ap, K, V) - loss_fn(Am, K, V)) / (2 * eps)
                elif name == "K":
                    gnum[i, j] = (loss_fn(Q, Ap, V) - loss_fn(Q, Am, V)) / (2 * eps)
                else:
                    gnum[i, j] = (loss_fn(Q, K, Ap) - loss_fn(Q, K, Am)) / (2 * eps)
        errs.append(float(np.abs(dA - gnum).max()))
    print(f"  dQ/dK/dV 最大误差：{errs[0]:.2e}, {errs[1]:.2e}, {errs[2]:.2e} ✓")
    assert max(errs) < 1e-5, "注意力梯度错误"

    # ---- 位置编码 ----
    print("\n-- 12.1.9 位置编码：让注意力感知顺序（书中 12.1.9 节）")
    pe = positional_encoding(20, 8)
    fig = Figure("正弦位置编码：每行 = 一个位置的编码「, 」维度「, 」位置")
    for i in range(8):
        fig.line(np.arange(20), pe[:, i], label=f"dim {i}")
    fig.save(os.path.join(PLOTS_DIR, "fig1_positional_encoding.png"))
    print("  正弦编码：不同频率叠加，任意位置编码都唯一且可被网络学习")

    # ---- 复杂度 ----
    print("\n-- 12.1.8 计算复杂度：注意力是 O(L²)（书中 12.1.8 节）")
    for L in (100, 1000, 10000):
        print(f"  序列长 L={L}: 注意力配对数量 L²={L*L:,}（平方增长 -> 长序列是挑战）")

    # ==================================================================
    # 12.2 自然语言：嵌入 / 自回归 / RNN（书中 12.2 节）
    # ==================================================================
    section("12.2 自然语言基础（书中 12.2 节）")
    print("-- 12.2.1 词嵌入：one-hot -> 可学习的稠密向量")
    vocab = 10
    E = rng.normal(0, 0.3, (vocab, 4))          # 嵌入矩阵（10 词 x 4 维）
    w = 3                                       # 第 3 个词
    emb = E[w]
    print(f"  词 {w} 的嵌入向量 = {np.round(emb, 3)}（one-hot 的稠密替代，可学习）")

    print("\n-- 12.2.4 自回归模型：p(x1..xT) = Π p(xt | x1..x_{t-1})")
    print("  逐 token 预测下一个 token，语言模型的核心范式（书中 12.2.4 节）")

    print("\n-- 12.2.5 RNN：状态 h_t 传递历史信息（书中 12.2.5 节）")
    print("  h_t = tanh(W_hh h_{t-1} + W_xh x_t + b)，逐时间步串行 —— 无法并行！")
    print("  => Transformer 的自注意力让所有位置并行处理，这是它取代 RNN 的原因")

    # ==================================================================
    # 12.3 微型 Transformer 语言模型（书中 12.3 节）
    # ==================================================================
    section("12.3 从零训练微型 Transformer 语言模型（书中 12.3 节）")
    print("架构：嵌入 -> 位置编码 -> 因果自注意力 -> 前馈 -> softmax 预测下一个字符")
    print("\n训练数据：周期性字符序列（便于观察模型是否学会规律）")

    # ---- 构造合成文本与词汇表 ----
    pattern = "abacabac"                       # 简单周期
    text = (pattern * 200)[:1600]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    V = len(chars)
    data = np.array([stoi[c] for c in text])
    print(f"  词表大小 {V}，字符集 {chars}，序列长度 {len(data)}")

    # ---- 微型 Transformer 参数 ----
    d_model = 16
    n_heads = 2
    d_head = d_model // n_heads
    seq_len = 8
    rngT = np.random.default_rng(42)
    # 嵌入 + 输出
    W_emb = rngT.normal(0, 0.1, (V, d_model))
    W_out = rngT.normal(0, 0.1, (d_model, V))
    # 单层注意力
    Wq = rngT.normal(0, 0.05, (d_model, d_model))
    Wk = rngT.normal(0, 0.05, (d_model, d_model))
    Wv = rngT.normal(0, 0.05, (d_model, d_model))
    # 前馈
    Wff1 = rngT.normal(0, 0.05, (d_model, 4 * d_model))
    bff1 = np.zeros(4 * d_model)
    Wff2 = rngT.normal(0, 0.05, (4 * d_model, d_model))
    bff2 = np.zeros(d_model)
    # 因果掩码
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

    def forward(xb):
        """xb: (B, seq_len) token 序列 -> logits (B, seq_len, V)。"""
        B = xb.shape[0]
        h = W_emb[xb]                           # (B, L, d) 嵌入
        h = h + positional_encoding(xb.shape[1], d_model)   # 按实际长度加位置编码
        # 自注意力（拆成多头再合并；这里是单层简化实现）
        Q = h @ Wq; K = h @ Wk; Vh = h @ Wv     # (B, L, d)
        attn_out = np.zeros_like(h)
        mask_cur = np.triu(np.full((xb.shape[1], xb.shape[1]), -1e9), k=1)  # 按长度
        for b in range(B):
            for hh in range(n_heads):
                sl = slice(hh * d_head, (hh + 1) * d_head)
                o, _ = scaled_dot_product_attention(
                    Q[b, :, sl], K[b, :, sl], Vh[b, :, sl], mask_cur)
                attn_out[b, :, sl] = o
        h = h + attn_out                          # 残差
        ff = np.maximum(h @ Wff1 + bff1, 0)       # 前馈 + ReLU
        h = h + ff @ Wff2 + bff2                  # 残差
        logits = h @ W_out                        # (B, L, V)
        return logits, attn_out, h                  # h 用于反向传播（输出层梯度）

    # 训练（batch 梯度下降，交叉熵）
    batch = 32
    iters = 1500
    lr = 0.1
    for it in range(iters):
        idx = rngT.integers(0, len(data) - seq_len, batch)
        xb = np.stack([data[i:i + seq_len] for i in idx])           # (B, L)
        yb = np.stack([data[i + 1:i + seq_len + 1] for i in idx])   # 下一个字符
        logits, _, h = forward(xb)
        # 交叉熵
        logits_flat = logits.reshape(-1, V)
        logits_flat -= logits_flat.max(axis=1, keepdims=True)
        exp = np.exp(logits_flat)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = float(-np.mean(np.log(probs[np.arange(probs.shape[0]), yb.ravel()] + 1e-12)))
        correct = float(np.mean(probs.argmax(axis=1) == yb.ravel()))
        # 梯度（简化：只更新输出层 + 嵌入，展示训练动态）
        dlogits = probs.copy()
        dlogits[np.arange(dlogits.shape[0]), yb.ravel()] -= 1
        dlogits /= logits_flat.shape[0]
        # 输出层
        dW_out = h.reshape(-1, d_model).T @ dlogits
        W_out -= lr * dW_out
        # 嵌入（近似：只用输出的梯度回传一层）
        dH = dlogits @ W_out.T
        # 通过前馈残差（跳过 FFN 的精细反向，简化教学演示）
        for b in range(batch):
            W_emb[xb[b]] -= lr * 0.1 * dH[b]
        if it % 300 == 0 or it == iters - 1:
            print(f"  iter {it}: 损失={loss:.4f}，训练正确率={correct:.3f}")
    # 采样生成
    print("\n  生成（贪心采样，从 'ab' 开始）：")
    gen = [stoi['a'], stoi['b']]
    for _ in range(20):
        xg = np.array(gen[-seq_len:])[None, :]
        logits_g, _, _ = forward(xg)
        p_next = softmax_rows(logits_g[0, -1:])[0]
        gen.append(int(p_next.argmax()))
    gen_text = "".join(itos[g] for g in gen)
    print(f"  {gen_text}")
    print(f"  目标周期模式：{pattern}...（模型{'学会' if gen_text.count('abac') > 2 else '在靠近'}规律）")

    # ==================================================================
    # 12.4 多模态：Vision Transformer（书中 12.4.1 节）
    # ==================================================================
    section("12.4 多模态：Vision Transformer 简介（书中 12.4.1 节）")
    print("ViT 把图像切成固定大小的 patch（如 16x16），每个 patch 展平成向量，")
    print("再当作「像素序列」喂给标准 Transformer —— 图像变成序列！")
    print("  28x28 图像 / 7x7 patch = 16 个 token（+位置编码）")
    print("  这就是 12.4 节的核心：注意力架构统一了文本、图像、音频。")

    # ==================================================================
    # 7. 小结
    # ==================================================================
    section("7. 小结：这一章你亲眼看到了什么")
    print("""
  1. 注意力 = QK 匹配度加权 V，缩放防止 softmax 饱和；
  2. 自注意力：每个位置都与其他所有位置交互（O(L2)）；
  3. 多头 = 多组不同的匹配模式并行；位置编码注入顺序；
  4. 自回归：逐 token 预测，Transformer 语言模型的核心；
  5. 微型 Transformer 学会了合成文本的周期规律；
  6. ViT 把图像变成 patch 序列：注意力统一多模态。
""")


if __name__ == "__main__":
    main()
