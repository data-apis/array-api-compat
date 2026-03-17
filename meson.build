project(
  'array_api_compat',
  version: '1.15.0.dev0',
  license: 'MIT',
  license_files: ['LICENSE']
)

py = import('python').find_installation()

sources = {
  'array_api_compat': files(
    'src/array_api_compat/__init__.py',
    'src/array_api_compat/_internal.py',
  ),

  'array_api_compat/common': files(
    'src/array_api_compat/common/__init__.py',
    'src/array_api_compat/common/_aliases.py',
    'src/array_api_compat/common/_fft.py',
    'src/array_api_compat/common/_helpers.py',
    'src/array_api_compat/common/_linalg.py',
    'src/array_api_compat/common/_typing.py',
  ),

  'array_api_compat/cupy': files(
    'src/array_api_compat/cupy/__init__.py',
    'src/array_api_compat/cupy/_aliases.py',
    'src/array_api_compat/cupy/_info.py',
    'src/array_api_compat/cupy/_typing.py',
    'src/array_api_compat/cupy/fft.py',
    'src/array_api_compat/cupy/linalg.py',
  ),

  'array_api_compat/dask': files(
    'src/array_api_compat/dask/__init__.py',
  ),

  'array_api_compat/dask/array': files(
    'src/array_api_compat/dask/array/__init__.py',
    'src/array_api_compat/dask/array/_aliases.py',
    'src/array_api_compat/dask/array/_info.py',
    'src/array_api_compat/dask/array/fft.py',
    'src/array_api_compat/dask/array/linalg.py',
  ),

  'array_api_compat/numpy': files(
    'src/array_api_compat/numpy/__init__.py',
    'src/array_api_compat/numpy/_aliases.py',
    'src/array_api_compat/numpy/_info.py',
    'src/array_api_compat/numpy/_typing.py',
    'src/array_api_compat/numpy/fft.py',
    'src/array_api_compat/numpy/linalg.py',
  ),

  'array_api_compat/torch': files(
    'src/array_api_compat/torch/__init__.py',
    'src/array_api_compat/torch/_aliases.py',
    'src/array_api_compat/torch/_info.py',
    'src/array_api_compat/torch/_typing.py',
    'src/array_api_compat/torch/fft.py',
    'src/array_api_compat/torch/linalg.py',
  ),
}

foreach subdir, files : sources
  py.install_sources(files, subdir: subdir)
endforeach

subdir('tests')
subdir('vendor_test')
