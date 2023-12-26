"""
A little CI shim for the dask backend that
disables the dask scheduler
"""
import dask
dask.config.set(scheduler='synchronous')

from dask.distributed import Client
_client = Client()
print(_client.dashboard_link)

from .dask import *
from .dask import __array_api_version__
