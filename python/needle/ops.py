"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from .autograd import as_tuple
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        return (self.scalar * power_scalar(node.inputs[0], self.scalar - 1) * out_grad, )


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad / rhs, -1 * out_grad * lhs / power_scalar(rhs, 2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar, )


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is not None:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            dim = len(a.shape)
            return array_api.swapaxes(a, dim -2, dim -1)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        # todo reverse reshape
        return reshape(out_grad, node.inputs[0].shape)

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # (3, 1) => (3, 3)
        # (1, 3) => (3, 3)
        # (1,) => (3, 3, 3)
        # () => (3, 3, 3)
        # (5, 4, 1) => (5, 4, 3)
        # (2,3) => (4,2,3) not ok for (2, 3, 4)
        shape_in = node.inputs[0].shape
        shape_out = out_grad.shape
        # 只能在最前面加维度，后面只能做1->x的提升
        # 分两步，一步是新增维度做sum，去除axis
        # 第二步是保留dim的sum
        # print("shape_in:\t", shape_in)
        # print("shape_out:\t", shape_out)
        if len(shape_in) != len(shape_out):
            out_grad = summation(out_grad, tuple(i for i in range(len(shape_out) - len(shape_in))))
        axes = []
        for i, dim in enumerate(shape_in):
            if dim == 1:
                axes.append(i)
        # print("axes:\t", axes)
        return summation(out_grad, tuple(axes)).reshape(shape_in)



def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        # (5, 4) (1, )
        # (5, 4) (0, )
        # (5, 4) (0, 1)
        # (5, 4, 1) (0, 1)
        shape_in = node.inputs[0].shape
        shape_out = out_grad.shape
        shape_tmp = list(shape_in)
        if self.axes is not None:
            for ax in as_tuple(self.axes):
                shape_tmp[ax] = 1
        else:
            shape_tmp = [1 for _ in shape_in]
        # print("shape_in:\t", shape_in)
        # print("shape_out:\t", shape_out)
        # print("shape_tmp:\t", shape_tmp)
        return broadcast_to(out_grad.reshape(shape_tmp), shape_in)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # m x n, n x k  ==> m x k
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        if len(lhs_grad.shape) != len(lhs.shape):
            lhs_grad = summation(lhs_grad, axes=tuple(i for i in range(len(lhs_grad.shape) - len(lhs.shape))))
        if len(rhs_grad.shape) != len(rhs.shape):
            rhs_grad = summation(rhs_grad, axes=tuple(i for i in range(len(rhs_grad.shape) - len(rhs.shape))))
        return lhs_grad, rhs_grad
        # b1 x b2 x m x n, b1 x b2 x n x k => b1 x b2 x m x k
        # b1 x b2 x m x n, n x k => b1 x b2 x m x k


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return Tensor(array_api.where(node.inputs[0].realize_cached_data() > 0, out_grad.realize_cached_data(), 0))

def relu(a):
    return ReLU()(a)

