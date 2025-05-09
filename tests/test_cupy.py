import pytest
from array_api_compat import device, to_device

xp = pytest.importorskip("array_api_compat.cupy")
from cupy.cuda import Stream


def test_to_device_with_stream():
    devices = xp.__array_namespace_info__().devices()
    streams = [
        Stream(),
        Stream(non_blocking=True), 
        Stream(null=True),
        Stream(ptds=True), 
        123,  # dlpack stream
    ]

    a = xp.asarray([1, 2, 3])
    for dev in devices:
        for stream in streams:
            b = to_device(a, dev, stream=stream)
            assert device(b) == dev
