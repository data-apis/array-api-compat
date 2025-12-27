from numpy.testing import assert_equal
import pytest

from array_api_compat import (
    device,
    to_device,
    is_jax_array,
    is_lazy_array,
    is_array_api_obj,
    is_writeable_array,
)

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pytestmark = pytest.skip(allow_module_level=True, reason="jax not found")

HAS_JAX_0_4_31 = jax.__version__ >= "0.4.31"


@pytest.mark.parametrize(
    "func",
    [
        lambda x: jnp.zeros(1, device=device(x)),
        lambda x: jnp.zeros_like(jnp.ones(1, device=device(x))),
        lambda x: jnp.zeros_like(jnp.empty(1, device=device(x))),
        lambda x: jnp.full(1, fill_value=0, device=device(x)),
        pytest.param(
            lambda x: jnp.asarray([0], device=device(x)),
            marks=pytest.mark.skipif(
                not HAS_JAX_0_4_31, reason="asarray() has no device= parameter"
            ),
        ),
        lambda x: to_device(jnp.zeros(1), device(x)),
    ],
)
def test_device_jit(func):
    # Test work around to https://github.com/jax-ml/jax/issues/26000
    # Also test missing to_device() method in JAX < 0.4.31
    # when inside jax.jit, even after importing jax.experimental.array_api

    x = jnp.ones(1)
    assert_equal(func(x), jnp.asarray([0]))
    assert_equal(jax.jit(func)(x), jnp.asarray([0]))


def test_inside_jit():
    # Test if jax arrays are handled correctly inside jax.jit.
    # Jax tracers are not a subclass of jax.Array from 0.8.2 on. We explicitly test that
    # tracers are handled appropriately. For limitations, see is_jax_array() docstring.
    # Reference issue: https://github.com/data-apis/array-api-compat/issues/368
    x = jnp.asarray([1, 2, 3])
    assert jax.jit(is_jax_array)(x)
    assert jax.jit(is_array_api_obj)(x)
    assert not jax.jit(is_writeable_array)(x)
    assert jax.jit(is_lazy_array)(x)
