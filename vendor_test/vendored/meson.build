py.install_sources([
    '__init__.py',
  ],
  subdir: 'array_api_compat/vendor_test/vendored'
)

# mimicking the symlink
# TODO: this seems to fail for editable builds
foreach subdir, paths : sources_raw
  foreach path : paths
    filename = path.split('/')[-1]
    vendor_subdir = subdir.replace('array_api_compat/', 'array_api_compat/vendor_test/vendored/_compat/')
    vendor_file = configure_file(input: '../..' / path, output: filename, copy: true, build_subdir: subdir, install_dir: py.get_install_dir() / vendor_subdir)
  endforeach
endforeach
