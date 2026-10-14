###############################################################################
#  BreathScheduler — 呼吸/肩动调度（P0：3–5s 正弦，body dy 偏移 1–3 px）
#
#  直接在合成阶段对 body_frame 做垂直平移（np.roll / cv2.warpAffine 皆可）。
#  零 GPU 推理；单帧成本 ~O(H*W)。
###############################################################################

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class BreathScheduler:
    """参数：
        period_s: 呼吸周期（默认 4s，范围 3–5）
        amplitude_px: 峰值偏移（默认 2 px，范围 1–3）
    """

    def __init__(self, period_s: float = 4.0, amplitude_px: float = 2.0):
        self.period = float(period_s)
        self.amp = float(amplitude_px)
        self._t0 = time.monotonic()

    def dy_offset(self, now: Optional[float] = None) -> float:
        """返回当前帧 dy 偏移（浮点 px，正=向下）。"""
        if now is None:
            now = time.monotonic()
        phase = 2.0 * math.pi * ((now - self._t0) / self.period)
        return self.amp * math.sin(phase)

    def apply(self, frame: NDArray[np.uint8], ctx: dict) -> NDArray[np.uint8]:
        """对整帧做垂直平移（顶部保留边缘像素，避免黑色边条）。

        约定：ctx 需含 'now'。P0：即使 enable_breath=1，如果 amplitude=0 也 pass-through。
        """
        dy = self.dy_offset(ctx.get("now"))
        ctx["breath_dy"] = dy
        if abs(dy) < 0.5:
            return frame
        return self._shift_y(frame, dy)

    @staticmethod
    def _shift_y(frame: NDArray[np.uint8], dy: float) -> NDArray[np.uint8]:
        """整数化 dy 平移；顶部/底部用边缘像素延展，避免黑边。"""
        import cv2
        h, w = frame.shape[:2]
        shift = int(round(dy))
        if shift == 0:
            return frame
        # cv2.translate 等价实现
        M = np.float32([[1, 0, 0], [0, 1, shift]])
        shifted = cv2.warpAffine(
            frame, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_LINEAR,
        )
        return shifted
