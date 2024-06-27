from jax.numpy import (
    # Constants
    e,
    inf,
    nan,
    pi,
    newaxis,
    # Dtypes
    bool,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    complex64,
    complex128,
    iinfo,
    finfo,
    can_cast,
    result_type,
    # functions
    zeros,
    all,
    any,
    isnan,
    isfinite,
    reshape
)
from jax.numpy import (
    asarray,
    s_,
    int_,
    argpartition,
    take_along_axis
)


def top_k(
    x,
    k,
    /,
    axis=None,
    *,
    largest=True,
):
    # The largest keyword can't be implemented with `jax.lax.top_k`
    # efficiently so am using `jax.numpy` for now
    if k <= 0:
        raise ValueError(f'k(={k}) provided must be positive.')

    positive_axis: int
    _arr = asarray(x)
    if axis is None:
        arr = _arr.ravel()
        positive_axis = 0
    else:
        arr = _arr
        positive_axis = axis if axis > 0 else axis % arr.ndim

    slice_start = (s_[:],) * positive_axis
    if largest:
        indices_array = argpartition(arr, -k, axis=axis)
        slice = slice_start + (s_[-k:],)
        topk_indices = indices_array[slice]
    else:
        indices_array = argpartition(arr, k-1, axis=axis)
        slice = slice_start + (s_[:k],)
        topk_indices = indices_array[slice]

    topk_indices = topk_indices.astype(int_)
    topk_values = take_along_axis(arr, topk_indices, axis=axis)
    return (topk_values, topk_indices)


__all__ = ['top_k', 'e', 'inf', 'nan', 'pi', 'newaxis', 'bool',
           'float32', 'float64', 'int8', 'int16', 'int32',
           'int64', 'uint8', 'uint16', 'uint32', 'uint64',
           'complex64', 'complex128', 'iinfo', 'finfo',
           'can_cast', 'result_type', 'zeros', 'all', 'isnan',
           'isfinite', 'reshape', 'any']
