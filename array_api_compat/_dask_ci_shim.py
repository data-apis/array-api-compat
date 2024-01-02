"""
A little CI shim for the dask backend that
disables the dask scheduler

It also lets you see the dask dashboard for debugging
at http://127.0.0.1:8787/status
"""
import dask
dask.config.set(scheduler='synchronous')

from dask.distributed import Client
_client = Client()

from .dask import *
from .dask import __array_api_version__
