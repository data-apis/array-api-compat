from numpy.testing import assert_equal
import pytest

from array_api_compat import device, to_device

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
    ]
)
def test_device_jit(func):
    # Test work around to https://github.com/jax-ml/jax/issues/26000
    # Also test missing to_device() method in JAX < 0.4.31
    # when inside jax.jit, even after importing jax.experimental.array_api

    x = jnp.ones(1)
    assert_equal(func(x), jnp.asarray([0]))
    assert_equal(jax.jit(func)(x), jnp.asarray([0]))
