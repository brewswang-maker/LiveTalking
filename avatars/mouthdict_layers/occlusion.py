###############################################################################
#  OcclusionDetector — 手遮挡嘴部区域检测（P0：帧差 + 肤色阈值命中 → 跳过后续层）
#
#  命中时 LayerCompositor 会跳过 mouth/eyes/emotion pass，只保留 body。
#  与记忆 df8712c8（bbox 抖动 / 手搭嘴）同源问题。
#
#  P2 会补多帧投票（3-of-5）与双阈值 hysteresis，此处先做单帧版本。
###############################################################################

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


# YCrCb 肤色阈值（经验值，覆盖亚洲/欧美肤色；P2 可细化）
_YCC_CB_LO = 77
_YCC_CB_HI = 127
_YCC_CR_LO = 133
_YCC_CR_HI = 173


class OcclusionDetector:
    """参数：
        skin_ratio_threshold: 肤色像素占比阈值（默认 0.25，超过则判为遮挡）
        diff_threshold: 帧差均值阈值（默认 15.0，避免静止肤色误判）
    """

    def __init__(
        self,
        skin_ratio_threshold: float = 0.25,
        diff_threshold: float = 15.0,
    ):
        self.skin_thr = float(skin_ratio_threshold)
        self.diff_thr = float(diff_threshold)
        self._prev_mouth_patch: Optional[NDArray[np.uint8]] = None

    def detect(self, frame: NDArray[np.uint8], mouth_bbox: tuple[int, int, int, int]) -> bool:
        """返回 True 表示该帧 mouth_bbox 疑似被手遮挡。

        frame: [H,W,3] uint8 BGR；mouth_bbox: (y1,y2,x1,x2)
        """
        import cv2
        y1, y2, x1, x2 = mouth_bbox
        if y2 <= y1 or x2 <= x1:
            return False
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return False

        # 帧差过滤：与上一帧同区域差异过小 → 忽略（避免静态背景肤色误命中）
        if self._prev_mouth_patch is not None and self._prev_mouth_patch.shape == patch.shape:
            diff = np.mean(np.abs(patch.astype(np.int16) - self._prev_mouth_patch.astype(np.int16)))
            if diff < self.diff_thr:
                self._prev_mouth_patch = patch.copy()
                return False
        self._prev_mouth_patch = patch.copy()

        ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        cb = ycc[:, :, 1]
        cr = ycc[:, :, 2]
        mask = (
            (cb >= _YCC_CB_LO) & (cb <= _YCC_CB_HI)
            & (cr >= _YCC_CR_LO) & (cr <= _YCC_CR_HI)
        )
        ratio = float(np.mean(mask))
        return ratio > self.skin_thr

    def reset(self) -> None:
        self._prev_mouth_patch = None
