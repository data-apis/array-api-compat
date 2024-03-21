# Releasing

To release, first make sure that all CI tests are passing on `main`.

Note that CuPy must be tested manually (it isn't tested on CI). Use the script

```
./test_cupy.sh
```

on a machine with a CUDA GPU.

Once you are ready to release, create a PR with a release branch, so that you
can verify that CI is passing. You must edit

```
array_api_compat/__init__.py
```

and update the version (the version is not computed from the tag because that
would break vendorability). You should also edit

```
docs/changelog.md
```

with the changes for the release.

Once everything is ready, create a tag

```
git tag -a <version>
```

(note the tag names are not prefixed, for instance, the tag for version 1.5 is
just `1.5`)

and push it to GitHub

```
git push origin <version>
```

Check that the `publish distributions` action on the tag build works. Note
that this action will run even if the other CI fails, so you must make sure
that CI is passing *before* tagging.

This does mean you can ignore CI failures, but ideally you should fix any
failures or update the `*-xfails.txt` files before tagging, so that CI and the
cupy tests pass. Otherwise it will be hard to tell what things are breaking in
the future. It's also a good idea to remove any xpasses from those files (but
be aware that some xfails are from flaky failures, so unless you know the
underlying issue has been fixed, an xpass test is probably still xfail).

If the publish action fails for some reason and didn't upload the release to
PyPI, you will need to delete the tag and try again.

After the PyPI package is published, the conda-forge bot should update the
feedstock automatically.
