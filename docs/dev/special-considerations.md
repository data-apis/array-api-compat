# Special Considerations

array-api-compat requires some special development considerations that are
different from most other Python libraries. The goal of array-api-compat is to
be a small library that packages can either vendor or add as a dependency to
implement array API support. Consequently, certain design considerations
should be taken into account:

(no-dependencies)=
- **No Hard Dependencies.** Although array-api-compat "depends" on NumPy, CuPy,
  PyTorch, etc., it does not hard depend on them. These libraries are not
  imported unless either an array object is passed to
  {func}`~.array_namespace()`, or the specific `array_api_compat.<namespace>`
  sub-namespace is explicitly imported. This is tested (as best as possible)
  in `tests/test_no_dependencies.py`.

- **Vendorability.** array-api-compat should be [vendorable](vendoring). This
  means that, for instance, all imports in the library are relative imports.
  No code in the package specifically references the name `array_api_compat`
  (we also support renaming the package to something else).
  Vendorability support is tested in `tests/test_vendoring.py`.

- **Pure Python.** To make array-api-compat as easy as possible to add as a
  dependency, the code is all pure Python.

- **Minimal Wrapping Only.** The wrapping functionality is minimal. This means
  that if something is difficult to wrap using pure Python, or if trying to
  support some array API behavior would require a significant amount of code,
  we prefer to leave the behavior as an upstream issue for the array library,
  and [document it as a known difference](../supported-array-libraries.md).

  This also means that we do not at this point in time implement anything
  other than wrappers for functions in the standard, and basic [helper
  functions](../helper-functions.rst) that would be useful for most users of
  array-api-compat. The addition of functions that are not part of the array
  API standard is currently out-of-scope for this package (see the
  [Scope](scope) section of the documentation).

- **No Side-Effects**. array-api-compat behavior should be localized to only the
  specific code that imports and uses it. It should be invisible to end-users
  or users of dependent codes. This in particular implies to the next two
  points.

- **No Monkey Patching.** `array-api-compat` should not attempt to modify
  anything about the underlying library. It is a *wrapper* library only.

- **No Modifying the Array Object.** The array (or tensor) object of the array
  library cannot be modified. This also precludes the creation of array
  subclasses or wrapper classes.

  Any non-standard behavior that is built-in to the array object, such as the
  behavior of [array
  methods](https://data-apis.org/array-api/latest/API_specification/array_object.html),
  is therefore left unwrapped. Users can workaround issues by using
  corresponding [elementwise
  functions](https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html)
  instead of
  [operators](https://data-apis.org/array-api/latest/API_specification/array_object.html#operators),
  and by using the [helper functions](../helper-functions.rst) provided by
  array-api-compat instead of attributes or methods like `x.to_device()`.

- **Avoid Restricting Behavior that is Outside the Scope of the Standard.** All
  array libraries have functions and behaviors that are outside of the scope
  of what is specified by the standard. These behaviors should be left intact
  whenever possible, unless the standard explicitly disallows something. This
  means

  - All namespaces are *extended* with wrapper functions. You may notice the
    extensive use of `import *` in various files in `array_api_compat`. While
    this would normally be questionable, this is the [one actual legitimate
    use-case for `import *`](https://peps.python.org/pep-0008/#imports), to
    re-export names from an external namespace.

  - All wrapper functions pass `**kwargs` through to the wrapped function.

  - Input types not supported by the standard should work if they work in the
    underlying wrapped function (for instance, Python scalars or `np.ndarray`
    subclasses).

  By keeping underlying behaviors intact, it is easier for libraries to swap
  out NumPy or other array libraries for array-api-compat, and it is easier
  for libraries to write array library-specific code paths.

  The onus is on users of array-api-compat to ensure their array API code is
  portable, e.g., by testing against [array-api-strict](array-api-strict).
