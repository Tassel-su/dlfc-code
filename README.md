# dlfc-code

把《Deep Learning: Foundations and Concepts》（Bishop & Bishop, Springer 2023）
**逐章转化为可运行、可验证、带中文讲解的 Python 代码**的学习仓库。

## 目标

不是"调 API"，而是**把书上的每一个公式亲手翻译成 numpy 代码**：
- 每章一个自包含脚本，从上到下顺序运行，中文叙述每步在做什么
- 每个核心公式配数值验证（闭式解对照 / 有限差分梯度检验 / assert）
- 每章配图（_plots/）、README（概念→公式→代码对照表 + 中文推导 + 自测题）
- 全程纯 numpy 从零实现（第 8 章起使用自研迷你自动微分引擎）

## 章节进度

| 章 | 主题 | 状态 |
|---|---|---|
| 1 | 深度学习革命（多项式曲线拟合） | ✅ 已完成 |
| 2 | 概率论 | ✅ 已完成（贝叶斯/高斯/信息论，含 10 万人生成筛查模拟） |
| 3 | 标准分布 | ✅ 已完成（多元高斯/共轭贝叶斯/von Mises/指数族/非参数） |
| 4 | 单层网络：回归 | ✅ 已完成（MLE=最小二乘/顺序学习/偏差-方差权衡） |
| 5 | 单层网络：分类 | ✅ 已完成（逻辑回归/Softmax/ROC/生成式判别式） |
| 6 | 深度神经网络 | ✅ 已完成（万能逼近/激活函数/表示学习/MDN） |
| 7 | 梯度下降 | ✅ 已完成（批量/SGD/动量/Adam/调度/归一化） |
| 8 | 反向传播（+迷你 autograd 引擎） | ✅ 已完成 |
| 9 | 正则化 | ✅ 已完成（L1/L2/早停/双重下降/残差/Dropout） |
| 10 | 卷积网络（MNIST） | ✅ 已完成（im2col/从零训练 CNN/显著性/FGSM） |
| 11 | 结构化分布（图模型） | ✅ 已完成（条件独立/朴素贝叶斯/HMM） |
| 12 | Transformer（注意力） | ✅ 已完成（微型语言模型训练生成） |
| 13 | 图网络 | ✅ 已完成（GCN 半监督分类/GAT/过平滑） |
| 14 | 采样（MCMC 等） | ✅ 已完成（拒绝/重要性/MH/Gibbs/Langevin） |
| 15 | 离散潜变量（K-means/GMM/EM） | ✅ 已完成（ELBO 单调） |
| 16 | 连续潜变量（PCA/PPCA/ELBO） | ✅ 已完成（MNIST 2D/PPCA） |
| 17 | 生成对抗网络 GAN | ✅ 已完成（2D 对抗训练） |
| 18 | Normalizing Flows | ✅ 已完成（耦合层/精确密度） |
| 19 | 自编码器（VAE） | ✅ 已完成（线性AE=PCA/重参数化/KL退火） |
| 20 | 扩散模型 | ✅ 已完成（前向/噪声预测/反向生成/Score） |

## 如何运行

    run.bat                  # 运行第 1 章（默认）
    run.bat ch05             # 运行第 5 章
    view_figs.bat            # 浏览器查看当前章图集

或直接用 Python 运行：

    C:/Python314/python.exe chapters/ch02_probabilities/ch02_probabilities.py

## 环境

- Python 3.14（C:/Python314），需要 numpy、matplotlib（均已安装）
- 无需 PyTorch/TensorFlow —— 全部从零实现，学习效果最大化
- 每章脚本 CPU 运行 ≤ 2 分钟

## 学习建议

1. 先跑脚本，再对照书看公式
2. 改 seed / 超参数做实验（脚本里留了旋钮）
3. 做每章 README 末尾的自测题（费曼学习法）
