# -*- coding: utf-8 -*-
"""MNIST 纯 numpy 加载器（Ch10 用）。

数据来源：ossci-datasets.s3.amazonaws.com（PyTorch 官方镜像），
已下载并保存为 .npy 于 C:/Users/eric/Desktop/Deep Learning/mnist_data/。
"""
import os
import numpy as np

_DATA_DIR = r"C:/Users/eric/Desktop/Deep Learning/mnist_data"


def load_mnist():
    """返回 (X_train, y_train, X_test, y_test)，图像为 float32 [0,1]，形状 (N,28,28)。"""
    X_train = np.load(os.path.join(_DATA_DIR, "train_images.npy")).astype(np.float32) / 255.0
    y_train = np.load(os.path.join(_DATA_DIR, "train_labels.npy"))
    X_test = np.load(os.path.join(_DATA_DIR, "test_images.npy")).astype(np.float32) / 255.0
    y_test = np.load(os.path.join(_DATA_DIR, "test_labels.npy"))
    return X_train, y_train, X_test, y_test


def to_onehot(y, K=10):
    """标签 -> 1-of-K 编码。"""
    return np.eye(K)[y]
