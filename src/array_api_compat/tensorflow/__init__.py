from collections.abc import Sequence
from builtins import all as py_all
from builtins import any as py_any
from builtins import bool as py_bool
from builtins import int as py_int
from builtins import slice as py_slice
from builtins import tuple as py_tuple
from typing import Final

import tensorflow as _tf

from .._internal import clone_module

__all__ = clone_module("tensorflow", globals())

Sequence.register(type(_tf.TensorShape(())))


class _ArrayAPIShape(py_tuple):
    def __new__(cls, tensor_shape):
        obj = super().__new__(cls, py_tuple(tensor_shape.as_list()))
        obj._tensor_shape = tensor_shape
        return obj

    @property
    def rank(self):
        return self._tensor_shape.rank

    @property
    def ndims(self):
        return self._tensor_shape.ndims

    def as_list(self):
        return self._tensor_shape.as_list()

    def __getitem__(self, key):
        out = self._tensor_shape[key]
        if isinstance(key, py_slice):
            return type(self)(out)
        return out

    def __getattr__(self, name):
        return getattr(self._tensor_shape, name)


def _patch_eager_tensor_shape() -> None:
    eager_tensor = type(_tf.constant(0))
    if getattr(eager_tensor, "_array_api_compat_shape_patched", False):
        return
    old_shape = eager_tensor.shape

    def shape(self):
        return _ArrayAPIShape(old_shape.__get__(self, eager_tensor))

    eager_tensor.shape = property(shape)
    eager_tensor._array_api_compat_shape_patched = True


_patch_eager_tensor_shape()


def _patch_eager_tensor_getitem() -> None:
    eager_tensor = type(_tf.constant(0))
    if getattr(eager_tensor, "_array_api_compat_getitem_patched", False):
        return
    old_getitem = eager_tensor.__getitem__

    def tensor_index(key):
        if isinstance(key, _tf.Tensor):
            return int(key) if key.shape == () else key
        if isinstance(key, py_slice):
            return py_slice(tensor_index(key.start), tensor_index(key.stop), tensor_index(key.step))
        if isinstance(key, py_tuple):
            return py_tuple(tensor_index(k) for k in key)
        return key

    def bool_getitem(self, key):
        x_shape = py_tuple(self.shape)
        key_shape = py_tuple(key.shape)
        if key.ndim > self.ndim or py_any(
            ks not in (xs, 0) for xs, ks in zip(x_shape, key_shape, strict=False)
        ):
            raise IndexError("boolean index did not match indexed array")
        if key.ndim == 0:
            x = _tf.reshape(self, (1,) + x_shape)
            return _tf.boolean_mask(x, _tf.reshape(key, (1,)))
        if py_any(ks == 0 for ks in key_shape):
            return _tf.zeros((0,) + x_shape[key.ndim :], dtype=self.dtype)
        return _tf.boolean_mask(self, key)

    def high_dim_int_getitem(self, key):
        shape = py_tuple(self.shape)
        if len(shape) <= 7:
            return None
        key = tensor_index(key)
        if isinstance(key, py_int):
            key = (key,)
        if not isinstance(key, py_tuple) or len(key) != len(shape):
            return None
        if not py_all(isinstance(k, py_int) and not isinstance(k, py_bool) for k in key):
            return None
        normalized = []
        for k, side in zip(key, shape, strict=True):
            k = k + side if k < 0 else k
            if k < 0 or k >= side:
                raise IndexError("index out of bounds")
            normalized.append(k)
        flat_index = 0
        for k, stride in zip(normalized, _strides(shape), strict=True):
            flat_index += k * stride
        return _tf.gather(_tf.reshape(self, (-1,)), flat_index)

    def _strides(shape):
        strides = []
        stride = 1
        for side in reversed(shape):
            strides.append(stride)
            stride *= side
        return py_tuple(reversed(strides))

    def getitem(self, key):
        if isinstance(key, _tf.Tensor) and key.dtype == _tf.bool:
            return bool_getitem(self, key)
        out = high_dim_int_getitem(self, key)
        if out is not None:
            return out
        return old_getitem(self, tensor_index(key))

    eager_tensor.__getitem__ = getitem
    eager_tensor._array_api_compat_getitem_patched = True


_patch_eager_tensor_getitem()


def _patch_eager_tensor_binary_ops() -> None:
    eager_tensor = type(_tf.constant(0))
    if getattr(eager_tensor, "_array_api_compat_binary_ops_patched", False):
        return

    def wrap(name, func_name, *, reverse=False):
        func = globals()[func_name]

        def op(self, other):
            if reverse:
                return func(other, self)
            return func(self, other)

        setattr(eager_tensor, name, op)

    def wrap_unary(name, func_name):
        func = globals()[func_name]

        def op(self):
            return func(self)

        setattr(eager_tensor, name, op)

    for name, func_name, reverse in [
        ("__add__", "add", False),
        ("__radd__", "add", True),
        ("__sub__", "subtract", False),
        ("__rsub__", "subtract", True),
        ("__mul__", "multiply", False),
        ("__rmul__", "multiply", True),
        ("__floordiv__", "floor_divide", False),
        ("__rfloordiv__", "floor_divide", True),
        ("__mod__", "remainder", False),
        ("__rmod__", "remainder", True),
        ("__pow__", "pow", False),
        ("__rpow__", "pow", True),
        ("__lt__", "less", False),
        ("__le__", "less_equal", False),
        ("__gt__", "greater", False),
        ("__ge__", "greater_equal", False),
        ("__eq__", "equal", False),
        ("__ne__", "not_equal", False),
        ("__and__", "bitwise_and", False),
        ("__rand__", "bitwise_and", True),
        ("__or__", "bitwise_or", False),
        ("__ror__", "bitwise_or", True),
        ("__xor__", "bitwise_xor", False),
        ("__rxor__", "bitwise_xor", True),
        ("__lshift__", "bitwise_left_shift", False),
        ("__rlshift__", "bitwise_left_shift", True),
        ("__rshift__", "bitwise_right_shift", False),
        ("__rrshift__", "bitwise_right_shift", True),
        ("__matmul__", "matmul", False),
        ("__rmatmul__", "matmul", True),
    ]:
        wrap(name, func_name, reverse=reverse)

    for name, func_name in [
        ("__abs__", "abs"),
        ("__invert__", "bitwise_invert"),
        ("__neg__", "negative"),
        ("__pos__", "positive"),
    ]:
        wrap_unary(name, func_name)

    eager_tensor._array_api_compat_binary_ops_patched = True


def _patch_eager_tensor_dlpack() -> None:
    eager_tensor = type(_tf.constant(0))
    if getattr(eager_tensor, "_array_api_compat_dlpack_patched", False):
        return
    old_dlpack = eager_tensor.__dlpack__

    def dlpack(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        tensor = _tf.identity(self) if copy is True else self
        del dl_device
        return old_dlpack(tensor, stream=stream, max_version=max_version)

    eager_tensor.__dlpack__ = dlpack
    eager_tensor._array_api_compat_dlpack_patched = True


# These imports may overwrite names from the import * above.
from . import _aliases
from ._aliases import *  # noqa: F403
from ._info import __array_namespace_info__  # noqa: F401

_patch_eager_tensor_binary_ops()
_patch_eager_tensor_dlpack()

__import__(__spec__.parent + ".linalg")
__import__(__spec__.parent + ".fft")

__array_api_version__: Final = "2025.12"

__all__ = sorted(
    set(__all__)
    | set(_aliases.__all__)
    | {"__array_api_version__", "__array_namespace_info__", "linalg", "fft"}
)


def __dir__() -> list[str]:
    return __all__
