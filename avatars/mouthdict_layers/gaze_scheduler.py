###############################################################################
#  GazeScheduler — 视线偏移调度器（P0 骨架：缓动 ±5° + 微眼跳 1–2 次/秒）
#
#  P0：无视线帧素材 → apply() 返回原帧，只输出 gaze_offset 供 P1 使用。
#  P1：按 4 方向（left/right/up/down/center）选近邻帧叠加。
###############################################################################

from __future__ import annotations

import math
import random
import time
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class GazeScheduler:
    """视线随目标点（鼠标坐标或画面中心）缓动，限幅 ±max_angle_deg；叠加微眼跳。

    参数：
        smoothing_alpha: 缓动系数（默认 0.1，即每帧向目标靠拢 10%）
        max_angle_deg: 视线偏移角限幅（默认 5°，超出经验值会显得斗鸡眼）
        micro_saccade_per_sec: 微眼跳频率（默认 1.5 次/秒）
        micro_saccade_amp_deg: 微眼跳幅度（默认 0.6°）
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.1,
        max_angle_deg: float = 5.0,
        micro_saccade_per_sec: float = 1.5,
        micro_saccade_amp_deg: float = 0.6,
        rng: Optional[random.Random] = None,
    ):
        self.alpha = float(smoothing_alpha)
        self.max_angle = float(max_angle_deg)
        self.micro_hz = float(micro_saccade_per_sec)
        self.micro_amp = float(micro_saccade_amp_deg)
        self._rng = rng or random.Random()

        # 当前偏移（度）
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        # 目标偏移（由 target 换算而来）
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        # 微眼跳瞬时偏移
        self._micro_x: float = 0.0
        self._micro_y: float = 0.0
        self._next_micro_at: float = time.monotonic() + self._next_micro_interval()

    # ---------------- 外部锚点 ----------------
    def set_target(self, target: Tuple[float, float], frame_size: Tuple[int, int]) -> None:
        """target 是画面上期望注视的点（像素坐标，例如鼠标位置），frame_size=(W,H)。

        把目标点相对画面中心的偏移归一化到 [-1,1]，再乘 max_angle 得目标视线角。
        """
        w, h = frame_size
        cx, cy = w / 2.0, h / 2.0
        if w <= 0 or h <= 0:
            return
        tx = (target[0] - cx) / (w / 2.0)
        ty = (target[1] - cy) / (h / 2.0)
        self._target_x = self._clamp(tx * self.max_angle)
        self._target_y = self._clamp(ty * self.max_angle)

    def _clamp(self, v: float) -> float:
        m = self.max_angle
        return max(-m, min(m, v))

    # ---------------- 状态推进 ----------------
    def _next_micro_interval(self) -> float:
        if self.micro_hz <= 0:
            return 1e9
        # 指数近似（泊松）
        return self._rng.expovariate(self.micro_hz)

    def update(self, now: Optional[float] = None) -> Tuple[float, float]:
        """每帧调用一次；返回最终 (offset_x_deg, offset_y_deg)。"""
        if now is None:
            now = time.monotonic()
        # 主缓动
        self._offset_x += self.alpha * (self._target_x - self._offset_x)
        self._offset_y += self.alpha * (self._target_y - self._offset_y)
        # 微眼跳
        if now >= self._next_micro_at:
            self._micro_x = self._rng.uniform(-self.micro_amp, self.micro_amp)
            self._micro_y = self._rng.uniform(-self.micro_amp, self.micro_amp)
            self._next_micro_at = now + self._next_micro_interval()
        else:
            # 微眼跳快速衰减（不是持久偏移）
            self._micro_x *= 0.6
            self._micro_y *= 0.6
        x = self._clamp(self._offset_x + self._micro_x)
        y = self._clamp(self._offset_y + self._micro_y)
        return x, y

    # ---------------- 合成 pass（P1 才生效；P0 无视线帧素材 → pass-through） ----------------
    def apply(self, frame: NDArray[np.uint8], ctx: dict) -> NDArray[np.uint8]:
        """约定 ctx 需含：'now'、可选 'mouse'（x,y 或 None）、'frame_size'、'gaze_frames'（dict[str, NDArray]）。

        P0：若无 gaze_frames → 只把最终 offset 写入 ctx['gaze_offset'] 供其它层用，返回原帧。
        """
        mouse = ctx.get("mouse")
        fsize = ctx.get("frame_size")
        if mouse is not None and fsize is not None:
            self.set_target(mouse, fsize)
        off_x, off_y = self.update(ctx.get("now"))
        ctx["gaze_offset"] = (off_x, off_y)

        gaze_frames = ctx.get("gaze_frames")
        eye_bbox = ctx.get("eye_bbox")
        if gaze_frames is None or eye_bbox is None:
            return frame

        # P1 落地：按主方向选近邻帧（4 方向 + center）
        # 角度阈值：|x|<0.5 且 |y|<0.5 → center
        key = self._pick_key(off_x, off_y)
        target = gaze_frames.get(key)
        if target is None:
            return frame
        y1, y2, x1, x2 = eye_bbox
        import cv2
        resized = cv2.resize(target, (x2 - x1, y2 - y1))
        out = frame.copy()
        out[y1:y2, x1:x2] = resized
        return out

    @staticmethod
    def _pick_key(x: float, y: float) -> str:
        if abs(x) < 0.5 and abs(y) < 0.5:
            return "center"
        if abs(x) >= abs(y):
            return "left" if x < 0 else "right"
        return "up" if y < 0 else "down"
