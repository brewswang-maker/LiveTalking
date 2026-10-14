###############################################################################
#  BlinkScheduler — 眨眼调度器（P0 骨架，泊松节律 + RMS 峰值抑制 + 句尾优先）
#
#  素材未交付时（P0）：eye_state 正常计算，apply() 无贴图可用 → 返回入参帧。
#  素材交付后（P1）：eye_state 选到眼帧图集 → 贴到 eye bbox。
###############################################################################

from __future__ import annotations

import random
import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from utils.logger import logger


# eye_state 枚举（避免依赖 enum.Enum 减少导入）
EYE_OPEN = "open"
EYE_CLOSING = "closing"
EYE_CLOSED = "closed"
EYE_OPENING = "opening"


class BlinkScheduler:
    """泊松节律眨眼调度。

    参数：
        blink_min_s / blink_max_s: 泊松间隔范围（默认 2–6s，符合人类 12–18 次/分）
        blink_curve_ms: 一次眨眼完整周期（闭-停-睁），默认 120ms
        rms_suppress_threshold: 平滑 RMS 高于此值时抑制眨眼（避免"大张口+闭眼"糊）
        suppress_cooldown_s: 抑制后至少间隔多久才允许下一次眨眼
    """

    def __init__(
        self,
        blink_min_s: float = 2.0,
        blink_max_s: float = 6.0,
        blink_curve_ms: int = 120,
        rms_suppress_threshold: float = 0.7,
        suppress_cooldown_s: float = 0.5,
        rng: Optional[random.Random] = None,
    ):
        self.blink_min_s = float(blink_min_s)
        self.blink_max_s = float(blink_max_s)
        self.blink_curve_ms = int(blink_curve_ms)
        self.rms_suppress = float(rms_suppress_threshold)
        self.cooldown = float(suppress_cooldown_s)
        self._rng = rng or random.Random()

        # 状态机
        self._next_blink_at: float = time.monotonic() + self._sample_interval()
        self._blink_start_at: Optional[float] = None
        self._suppressed_until: float = 0.0
        # 句读/停顿优先排入（外部调用 force_blink_soon 生效）
        self._pending_force_at: Optional[float] = None

        # 统计（用于验收：泊松间隔分布）
        self._blink_count: int = 0
        self._last_blink_at: float = 0.0

    # ---------------- 采样 ----------------
    def _sample_interval(self) -> float:
        """泊松近似：均匀落在 [min, max]，均值 ≈ (min+max)/2 ≈ 4s → 约 15 次/分。"""
        return self._rng.uniform(self.blink_min_s, self.blink_max_s)

    # ---------------- 外部锚点 ----------------
    def force_blink_soon(self, delay_s: float = 0.05) -> None:
        """句读/停顿点调用：把下一次眨眼提前到 now + delay_s（若原本更晚）。"""
        now = time.monotonic()
        target = now + max(0.0, float(delay_s))
        cur = self._pending_force_at
        if cur is None or target < cur:
            self._pending_force_at = target

    # ---------------- 状态推进 ----------------
    def update(self, *, now: Optional[float] = None, smoothed_rms: float = 0.0) -> str:
        """每帧调用一次；返回当前 eye_state ∈ {open, closing, closed, opening}。

        smoothed_rms: 来自 RmsToLevelMapper._smoothed（[0,1] 归一化能量），用于峰值抑制。
        """
        if now is None:
            now = time.monotonic()

        # 眨眼进行中 → 依时间返回 closing/closed/opening
        if self._blink_start_at is not None:
            elapsed_ms = (now - self._blink_start_at) * 1000.0
            half = self.blink_curve_ms / 2.0
            if elapsed_ms < half:
                return EYE_CLOSING
            if elapsed_ms < self.blink_curve_ms:
                return EYE_CLOSED
            # 完成
            self._blink_start_at = None
            self._next_blink_at = now + self._sample_interval()
            return EYE_OPENING

        # 峰值抑制：说话大张口时不眨
        if smoothed_rms > self.rms_suppress:
            self._suppressed_until = now + self.cooldown
            # 抑制期间把 next_blink_at 至少推到 cooldown 之后
            if self._next_blink_at < self._suppressed_until:
                self._next_blink_at = self._suppressed_until + self._sample_interval() * 0.5
            return EYE_OPEN

        # 强制眨眼（句尾）
        if self._pending_force_at is not None and now >= self._pending_force_at:
            self._pending_force_at = None
            self._start_blink(now)
            return EYE_CLOSING

        # 常规触发
        if now >= self._next_blink_at:
            self._start_blink(now)
            return EYE_CLOSING

        return EYE_OPEN

    def _start_blink(self, now: float) -> None:
        self._blink_start_at = now
        self._blink_count += 1
        self._last_blink_at = now

    # ---------------- 合成 pass（P1 才生效；P0 无 eye bbox / 眼帧素材 → pass-through） ----------------
    def apply(self, frame: NDArray[np.uint8], ctx: dict) -> NDArray[np.uint8]:
        """约定 ctx 需含：'now'（monotonic）、'smoothed_rms'（[0,1]）、可选 'eye_bbox'、'eye_frames'。

        P0：若 eye_bbox 或 eye_frames 缺失 → 只推进状态机、返回原帧。
        """
        eye_state = self.update(
            now=ctx.get("now"),
            smoothed_rms=float(ctx.get("smoothed_rms", 0.0)),
        )
        ctx["eye_state"] = eye_state

        eye_bbox = ctx.get("eye_bbox")
        eye_frames = ctx.get("eye_frames")
        if eye_bbox is None or eye_frames is None:
            return frame

        # P1 落地：按 eye_state 选帧贴到 eye_bbox（占位实现，等美术素材到位再启用）
        target = eye_frames.get(eye_state)
        if target is None:
            return frame
        y1, y2, x1, x2 = eye_bbox
        import cv2
        resized = cv2.resize(target, (x2 - x1, y2 - y1))
        out = frame.copy()
        out[y1:y2, x1:x2] = resized
        return out

    # ---------------- 观测 ----------------
    @property
    def blink_count(self) -> int:
        return self._blink_count
