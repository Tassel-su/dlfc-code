# -*- coding: utf-8 -*-
"""
迷你自动微分引擎（autograd.py）
================================
配套《Deep Learning: Foundations and Concepts》第 8 章（8.2 节）。

用途：
  用【反向模式自动微分】自动计算任意标量函数对全部输入的梯度，
  等价于第 8 章 8.2.2 节描述的 reverse-mode AD —— 也就是深度学习
  框架（PyTorch/TensorFlow）autograd 的微型实现。

核心思想：
  1. 每个 Value 节点记录：数据 data、梯度 grad、参与计算的子节点 _prev、
     局部求导规则 _backward；
  2. 前向：从输入开始逐节点计算，构建计算图（DAG）；
  3. 反向：拓扑排序后，从输出沿链式法则逐节点累加梯度（每个节点只算一次）。

本引擎只处理标量（实现简单、教学清晰）；要向量化只需把 data 换成
numpy 数组并保持同样结构 —— 原理完全相同。

示例：
    x = Value(2.0, label="x")
    y = (x * 2 + 1).tanh()          # 任意表达式
    y.backward()                    # 反向传播
    print(x.grad)                   # dy/dx
"""
from __future__ import annotations


class Value:
    """计算图中的一个节点：标量数据 + 梯度 + 局部反向规则。"""

    def __init__(self, data: float, _children=(), _op: str = "", label: str = ""):
        self.data = float(data)          # 该节点的数值（前向结果）
        self.grad = 0.0                  # 损失对该节点的梯度（反向时填充）
        self._backward = lambda: None    # 局部链式法则：把梯度传给子节点
        self._prev = set(_children)      # 参与计算本节点的子节点（图结构）
        self._op = _op                   # 产生本节点的运算（用于打印/调试）
        self.label = label               # 人类可读的名字

    # ---------- 运算与对应的局部梯度规则 ----------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # z = a + b  =>  dz/da = 1，dz/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # z = a * b  =>  dz/da = b，dz/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "指数必须是常量（简化版）"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # z = a^n  =>  dz/da = n * a^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(float(__import__("math").exp(self.data)), (self,), "exp")

        def _backward():
            # z = e^a  =>  dz/da = e^a = z（前向值刚好就是导数）
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(float(__import__("math").log(self.data)), (self,), "log")

        def _backward():
            # z = ln a  =>  dz/da = 1/a
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # z = tanh(a)  =>  dz/da = 1 - z²（tanh 导数的漂亮性质）
            self.grad += (1.0 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            # ReLU'(a) = 1 if a>0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ---------- 派生运算 ----------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        # a / b = a * b^(-1)
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ---------- 反向传播 ----------
    def backward(self):
        """从本节点开始反向传播：拓扑排序 + 逐节点链式法则。"""
        # 1) 拓扑排序：先处理所有"上游"节点（被依赖者优先）
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2) 输出节点梯度初始化为 1（d(输出)/d(输出) = 1）
        self.grad = 1.0
        # 3) 逆拓扑序逐节点把梯度传给子节点（链式法则）
        for v in reversed(topo):
            v._backward()

    # ---------- 打印 ----------
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
