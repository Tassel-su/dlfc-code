# 第 12 章：Transformer（注意力）

> 对应《Deep Learning: Foundations and Concepts》(Bishop & Bishop, Springer 2023)
> 第 12 章（印刷页 357-404，PDF 页 377-424）
> 脚本：`ch12_transformer.py`；输出图：`_plots/`

## 这一章在讲什么

注意力机制是深度学习的第二次革命：它让模型**直接比较序列中任意两个位置**
（无需 RNN 的串行依赖），可并行、可扩展。本章从零实现缩放点积注意力
（含反向传播与梯度验证）、多头注意力、位置编码，并训练一个微型
decoder Transformer 语言模型做字符级生成；最后简介 ViT（图像 patch 化）。

## 概念 → 公式 → 代码 对照表

| 书中概念 | 数学形式 | 代码位置 |
|---|---|---|
| 缩放点积注意力 | softmax(QKᵀ/√d)V | `scaled_dot_product_attention` |
| 注意力反向 | dQ,dK,dV（softmax 链式） | `attention_backward`（验证 <1e-5） |
| 多头注意力 | 多头并行 + 拼接 | 12.3 训练（n_heads=2） |
| 位置编码 | sin/cos 不同频率 | `positional_encoding` |
| 因果掩码 | 未来位置置 -inf | `mask_cur` |
| 自回归 | p(xt|历史) | 12.3 逐 token 生成 |
| 计算复杂度 | O(L²) | 12.1.8 |
| 词嵌入 | one-hot → 稠密向量 | 12.2.1 |
| RNN/BPTT | h_t = tanh(W·h_{t-1}+...) | 12.2.5（对比） |
| ViT | 图像 patch 序列化 | 12.4 |

## 运行结果（seed 固定，可直接复现）

- 注意力梯度验证：dQ/dK/dV 最大误差 ~1e-6 ✓
- 位置编码：20 位置 x 8 维正弦编码图
- **微型 Transformer 训练**（vocab=3, d=16, 1 层, 2 头）：1500 步后
  正确率 0.871，损失 0.587
- 生成："abacabac..."（学会部分周期规律）
- 复杂度：L=10000 时 L²=10⁸ 次配对

## 关键推导（中文）

### 推导 1：缩放点积注意力为什么除以 √d

score = q·k/√d。若 q,k 是 d 维零均值单位方差向量，q·k 的方差 ≈ d，
不缩放会让 softmax 的输入方差随 d 增大而饱和（梯度消失）。
除以 √d 使方差回到 O(1)，softmax 保持"有区分度"。

### 推导 2：注意力反向传播（核心三步）

设 a = softmax(QKᵀ/√d)，out = aV。对损失 L：
1. dV = aᵀ dout
2. da = dout Vᵀ（矩阵链式法则）
3. softmax 反向：ds = a ⊙ (da - Σ da⊙a 沿行)，再乘 √d 的倒数回 Q、K
代码 `attention_backward` 与数值梯度对照验证。

### 推导 3：为什么自注意力取代 RNN

RNN：h_t 依赖 h_{t-1}，串行计算，长序列梯度衰减（BPTT）。
自注意力：所有位置同时计算（矩阵运算），每个位置直接看到全部历史，
路径长度为 1（无信息丢失）。代价是 O(L²) 内存 —— 长序列是研究热点。

## 数值验证（脚本内置 assert）

1. 注意力权重行归一化（Σ=1）
2. 注意力反向 vs 数值梯度（< 1e-5）

## 自测题（答案在下方）

1. Q、K、V 分别是什么角色？
2. 为什么缩放？不缩放会怎样？
3. 因果掩码的作用？
4. 自回归生成时为什么只能看左侧？
5. ViT 如何把图像变成序列？

<details>
<summary>答案</summary>

1. Q（查询）= 当前"想知道什么"；K（键）= 其他位置"有什么"；
   V（值）= 其他位置"携带的信息"；注意力 = 用 QK 匹配度加权 V。
2. 不缩放时 q·k 的方差 ~d，softmax 输入过大趋于饱和（one-hot 化），
   梯度消失；除以 √d 保持 softmax 的区分度。
3. 语言模型只能看到当前位置及之前的 token（未来是未知的）；
   掩码把未来位置置 -inf，softmax 后权重为 0。
4. 自回归假设 p(x_t | x_1..x_{t-1})：预测时必须遮住未来，否则信息泄漏。
5. 把图像切成固定 patch（如 16x16），展平成向量加位置编码，
   当作"像素 token 序列"输入标准 Transformer。
</details>

## 如何运行

    run.bat ch12
