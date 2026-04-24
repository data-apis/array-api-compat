# Basic test that vendoring works

from .vendored._compat import (
    is_paddle_array,
    is_paddle_namespace,
    paddle as paddle_compat,
)

import paddle

def _test_paddle():
    a = paddle_compat.to_tensor([1., 2., 3.])
    b = paddle_compat.arange(3, dtype=paddle_compat.float64)
    assert a.dtype == paddle_compat.float32 == paddle.float32
    assert b.dtype == paddle_compat.float64 == paddle.float64

    # paddle.expand_dims does not exist. Update this to use something else if it is added
    res = paddle_compat.expand_dims(a, axis=0)
    assert res.dtype == paddle_compat.float32 == paddle.float32
    assert res.shape == [1, 3]
    assert isinstance(res.shape, list)
    assert isinstance(a, paddle.Tensor)
    assert isinstance(b, paddle.Tensor)
    assert isinstance(res, paddle.Tensor)

    assert paddle.allclose(res, paddle.to_tensor([[1., 2., 3.]]))

    assert is_paddle_array(res)
    assert is_paddle_namespace(paddle) and is_paddle_namespace(paddle_compat)

