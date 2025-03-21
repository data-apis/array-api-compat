# Tests

The majority of the behavior for array-api-compat is tested by the
[array-api-tests](https://github.com/data-apis/array-api-tests) test suite for
the array API standard. There are also array-api-compat specific tests in
[`tests/`](https://github.com/data-apis/array-api-compat/tree/main/tests).
These tests should be limited to things that are not tested by the test suite,
e.g., tests for [helper functions](../helper-functions.rst) or for behavior
that is not strictly required by the standard. To run these tests, install the
dependencies from the `dev` optional group (array-api-compat has [no hard
runtime dependencies](no-dependencies)).

array-api-tests is run against all supported libraries are tested on CI
([except for JAX](jax-support) and [Sparse](sparse-support)). This is achieved
by a [reusable GitHub Actions
Workflow](https://github.com/data-apis/array-api-compat/blob/main/.github/workflows/array-api-tests.yml).
Most libraries have tests that must be xfailed or skipped for various reasons.
These are defined in specific `<library>-xfails.txt` files and are
automatically forwarded to array-api-tests.

You may often need to update these xfail files, either to add new xfails
(e.g., because of new test suite features, or because a test that was
previously thought to be passing actually flaky fails). Try to keep the xfails
files organized, with comments pointing to upstream issues whenever possible.

From time to time, xpass tests should be removed from the xfail files, but be
aware that many xfail tests are flaky, so an xpass should only be removed if
you know that the underlying issue has been fixed.

Array libraries that require a GPU to run (currently only CuPy) cannot be
tested on CI. There is a helper script `test_cupy.sh` that can be used to
manually test CuPy on a machine with a CUDA GPU.
