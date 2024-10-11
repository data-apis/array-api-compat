from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

import array_api_compat

setup(
    name='array_api_compat',
    version=array_api_compat.__version__,
    packages=find_packages(include=["array_api_compat*"]),
    author="Consortium for Python Data API Standards",
    description="A wrapper around NumPy and other array libraries to make them compatible with the Array API standard",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://data-apis.org/array-api-compat/",
    license="MIT",
    extras_require={
        "numpy": "numpy",
        "cupy": "cupy",
        "jax": "jax",
        "pytorch": "pytorch",
        "dask": "dask",
        "sparse": "sparse >=0.15.1",
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
