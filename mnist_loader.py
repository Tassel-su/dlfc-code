# -*- coding: utf-8 -*-
"""MNIST 纯 numpy 加载器（Ch10/Ch11 共用）。

数据来源：ossci-datasets.s3.amazonaws.com（PyTorch 官方镜像）。
优先读取仓库内 data/mnist/；若不存在则自动下载（约 12MB）。
"""
import gzip
import os
import struct
import urllib.request

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DIR = os.path.join(_REPO_ROOT, "data", "mnist")
_LEGACY_DIR = r"C:/Users/eric/Desktop/Deep Learning/mnist_data"
_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train-images-idx3-ubyte.gz": "train_images.npy",
    "train-labels-idx1-ubyte.gz": "train_labels.npy",
    "t10k-images-idx3-ubyte.gz": "test_images.npy",
    "t10k-labels-idx1-ubyte.gz": "test_labels.npy",
}


def _read_idx(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:
            n, r, c = struct.unpack(">III", f.read(12))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
        n = struct.unpack(">I", f.read(4))[0]
        return np.frombuffer(f.read(), dtype=np.uint8)


def _ensure_data():
    for d in (_LOCAL_DIR, _LEGACY_DIR):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "train_images.npy")):
            return d
    os.makedirs(_LOCAL_DIR, exist_ok=True)
    print("[mnist_loader] 首次运行：正在下载 MNIST（约 12MB）...")
    for gz, out in _FILES.items():
        dest = os.path.join(_LOCAL_DIR, out)
        if os.path.exists(dest):
            continue
        tmp = os.path.join(_LOCAL_DIR, gz)
        data = urllib.request.urlopen(_BASE_URL + gz, timeout=300).read()
        with open(tmp, "wb") as f:
            f.write(data)
        arr = _read_idx(tmp)
        np.save(dest, arr)
        os.remove(tmp)
        print(f"  {out} {arr.shape}")
    return _LOCAL_DIR


_DATA_DIR = _ensure_data()


def load_mnist():
    X_train = np.load(os.path.join(_DATA_DIR, "train_images.npy")).astype(np.float32) / 255.0
    y_train = np.load(os.path.join(_DATA_DIR, "train_labels.npy"))
    X_test = np.load(os.path.join(_DATA_DIR, "test_images.npy")).astype(np.float32) / 255.0
    y_test = np.load(os.path.join(_DATA_DIR, "test_labels.npy"))
    return X_train, y_train, X_test, y_test


def to_onehot(y, K=10):
    return np.eye(K)[y]
