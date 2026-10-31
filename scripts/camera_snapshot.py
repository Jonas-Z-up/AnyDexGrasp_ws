"""Coherent copies of the existing RealSense RGB/depth shared buffers.

The producer publishes an odd/even sequence around BOTH writes. Host timestamp
is captured when the RGB-D pair is received, not when inference finishes.
This is a single-host snapshot protocol, not hardware pose synchronization.
"""
import time
import numpy as np

META_NAME = "realsense_frame_meta"
META_SIZE = 2 * np.dtype(np.int64).itemsize


def read_snapshot(depth_buffer, color_buffer, metadata, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        before = int(metadata[0])
        if before > 0 and before % 2 == 0:
            depth, color = depth_buffer.copy(), color_buffer.copy()
            captured_ns = int(metadata[1])
            after = int(metadata[0])
            if before == after and captured_ns > 0:
                return depth, color, {"frame_sequence": after, "captured_at": captured_ns / 1e9}
        time.sleep(0.002)
    raise RuntimeError("无法获得完整 RGB-D 快照，请检查或重启 realsense.py")
