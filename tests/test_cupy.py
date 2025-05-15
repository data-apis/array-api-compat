import pytest
from array_api_compat import device, to_device

xp = pytest.importorskip("array_api_compat.cupy")
from cupy.cuda import Stream


@pytest.mark.parametrize(
    "make_stream",
    [
        lambda: Stream(),
        lambda: Stream(non_blocking=True),
        lambda: Stream(null=True),
        lambda: Stream(ptds=True),
    ],
)
def test_to_device_with_stream(make_stream):
    devices = xp.__array_namespace_info__().devices()

    a = xp.asarray([1, 2, 3])
    for dev in devices:
        # Streams are device-specific and must be created within
        # the context of the device...
        with dev:
            stream = make_stream()
        # ... however, to_device() does not need to be inside the
        # device context.
        b = to_device(a, dev, stream=stream)
        assert device(b) == dev


def test_to_device_with_dlpack_stream():
    devices = xp.__array_namespace_info__().devices()

    a = xp.asarray([1, 2, 3])
    for dev in devices:
        # Streams are device-specific and must be created within
        # the context of the device...
        with dev:
            s1 = Stream()

        # ... however, to_device() does not need to be inside the
        # device context.
        b = to_device(a, dev, stream=s1.ptr)
        assert device(b) == dev
