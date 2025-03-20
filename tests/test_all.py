"""
Test that files that define __all__ aren't missing any exports.

You can add names that shouldn't be exported to _all_ignore, like

_all_ignore = ['sys']

This is preferable to del-ing the names as this will break any name that is
used inside of a function. Note that names starting with an underscore are automatically ignored.
"""


import sys

from ._helpers import import_, wrapped_libraries

import pytest

@pytest.mark.skip(reason="TODO: starts failing after adding test_torch.py in gh-277")
@pytest.mark.parametrize("library", ["common"] + wrapped_libraries)
def test_all(library):
    if library == "common":
        import array_api_compat.common  # noqa: F401
    else:
        import_(library, wrapper=True)

    for mod_name in sys.modules:
        if not mod_name.startswith('array_api_compat.' + library):
            continue

        module = sys.modules[mod_name]

        # TODO: We should define __all__ in the __init__.py files and test it
        # there too.
        if not hasattr(module, '__all__'):
            continue

        dir_names = [n for n in dir(module) if not n.startswith('_')]
        if '__array_namespace_info__' in dir(module):
            dir_names.append('__array_namespace_info__')
        ignore_all_names = getattr(module, '_all_ignore', [])
        ignore_all_names += ['annotations', 'TYPE_CHECKING']
        dir_names = set(dir_names) - set(ignore_all_names)
        all_names = module.__all__

        if set(dir_names) != set(all_names):
            extra_dir = set(dir_names) - set(all_names)
            extra_all = set(all_names) - set(dir_names)
            assert not extra_dir, f"Some dir() names not included in __all__ for {mod_name}: {extra_dir}"
            assert not extra_all, f"Some __all__ names not in dir() for {mod_name}: {extra_all}"
