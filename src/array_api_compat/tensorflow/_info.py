"""
Array API Inspection namespace for TensorFlow.
"""

from functools import cache

import tensorflow as tf


class __array_namespace_info__:
    """
    Get the array API inspection namespace for TensorFlow.
    """

    __module__ = "tensorflow"

    def capabilities(self):
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            "max dimensions": 254,
        }

    def default_device(self):
        return tf.constant(0).device

    def default_dtypes(self, *, device=None):
        return {
            "real floating": tf.float32,
            "complex floating": tf.complex64,
            "integral": tf.int32,
            "indexing": tf.int64,
        }

    def _dtypes(self, kind):
        bool = tf.bool
        int8 = tf.int8
        int16 = tf.int16
        int32 = tf.int32
        int64 = tf.int64
        uint8 = tf.uint8
        uint16 = tf.uint16
        uint32 = tf.uint32
        uint64 = tf.uint64
        float32 = tf.float32
        float64 = tf.float64
        complex64 = tf.complex64
        complex128 = tf.complex128

        if kind is None:
            return {
                "bool": bool,
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
                "uint8": uint8,
                "uint16": uint16,
                "uint32": uint32,
                "uint64": uint64,
                "float32": float32,
                "float64": float64,
                "complex64": complex64,
                "complex128": complex128,
            }
        if kind == "bool":
            return {"bool": bool}
        if kind == "signed integer":
            return {
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
            }
        if kind == "unsigned integer":
            return {
                "uint8": uint8,
                "uint16": uint16,
                "uint32": uint32,
                "uint64": uint64,
            }
        if kind == "integral":
            return {
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
                "uint8": uint8,
                "uint16": uint16,
                "uint32": uint32,
                "uint64": uint64,
            }
        if kind == "real floating":
            return {
                "float32": float32,
                "float64": float64,
            }
        if kind == "complex floating":
            return {
                "complex64": complex64,
                "complex128": complex128,
            }
        if kind == "numeric":
            return {
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
                "uint8": uint8,
                "uint16": uint16,
                "uint32": uint32,
                "uint64": uint64,
                "float32": float32,
                "float64": float64,
                "complex64": complex64,
                "complex128": complex128,
            }
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    @cache
    def dtypes(self, *, device=None, kind=None):
        res = self._dtypes(kind)
        for k, v in res.copy().items():
            try:
                with tf.device(device):
                    tf.zeros((), dtype=v)
            except Exception:
                del res[k]
        return res

    @cache
    def devices(self):
        devices = []
        for device in tf.config.list_logical_devices():
            with tf.device(device.name):
                devices.append(tf.constant(0).device)
        return tuple(dict.fromkeys(devices))
