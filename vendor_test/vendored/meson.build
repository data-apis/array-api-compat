py.install_sources([
    '__init__.py',
  ],
  subdir: 'array_api_compat/vendor_test/vendored'
)

# refers to the sources defined at the top level
foreach subdir, files : sources
  py.install_sources(
    files,
    subdir: subdir.replace('array_api_compat/', 'array_api_compat/vendor_test/vendored/_compat/'),
  )
endforeach

