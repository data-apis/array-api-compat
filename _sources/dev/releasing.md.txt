# Releasing

- [ ] **Create a PR with a release branch**

  This makes it easy to verify that CI is passing, and also gives you a place
  to push up updates to the changelog and any last minute fixes for the
  release.

- [ ] **Double check the release branch is fully merged with `main`.**

  (e.g., if the release branch is called `release`)

  ```
  git checkout main
  git pull
  git checkout release
  git merge main
  ```

- [ ] **Make sure that all CI tests are passing.**

  Note that the GitHub action that publishes to PyPI does not check if CI is
  passing before publishing. So you need to check this manually.

  This does mean you can ignore CI failures, but ideally you should fix any
  failures or update the `*-xfails.txt` files before tagging, so that CI and
  the CuPy tests fully pass. Otherwise it will be hard to tell what things are
  breaking in the future. It's also a good idea to remove any xpasses from
  those files (but be aware that some xfails are from flaky failures, so
  unless you know the underlying issue has been fixed, an xpass test is
  probably still xfail).

- [ ] **Test CuPy.**

  CuPy must be tested manually (it isn't tested on CI, see
  https://github.com/data-apis/array-api-compat/issues/197). Use the script

   ```
   ./test_cupy.sh
   ```

   on a machine with a CUDA GPU.


- [ ] **Update the version.**

  You must edit

  ```
  array_api_compat/__init__.py
  ```

  and update the version (the version is not computed from the tag because
  that would break vendorability).

- [ ] **Update the [changelog](../changelog.md).**

  Edit

  ```
  docs/changelog.md
  ```

  with the changes for the release.

- [ ] **Create the release tag.**

  Once everything is ready, create a tag

  ```
  git tag -a <version>
  ```

  (note the tag names are not prefixed, for instance, the tag for version 1.5 is
  just `1.5`)

- [ ] **Push the tag to GitHub.**

  *This is the final step. Doing this will build and publish the release!*

  ```
  git push origin <version>
  ```

  This will trigger the [`publish
  distributions`](https://github.com/data-apis/array-api-compat/actions/workflows/publish-package.yml)
  GitHub Action that will build the release and push it to PyPI.

- [ ] **Check that the [`publish
  distributions`](https://github.com/data-apis/array-api-compat/actions/workflows/publish-package.yml)
  action build on the tag worked.** Note that this action will run even if the
  other CI fails, so you must make sure that CI is passing *before* tagging.

  If it failed for some reason, you may need to delete the tag and try again.

- [ ] **Merge the release branch.**

  This way any changes you made in the branch, such as updates to the
  changelog or xfails files, are updated in `main`. This will also make the
  docs update (the docs are published automatically from the sources on
  `main`).

- [ ] **Update conda-forge.**

  After the PyPI package is published, the conda-forge bot should update the
  feedstock automatically after some time. The bot should automerge, so in
  most cases you don't need to do anything here, unless some metadata on the
  feedstock needs to be updated.
