###############################################################################
#  EmotionMood — 情绪状态机（P0：接口与 fade 计时；素材未到时只更新 state）
#
#  P1：情绪整帧图集 + cv2.addWeighted 200–300ms 淡入淡出。
#  P2：LLM 意图总线调用 set_mood 或 fade_to。
###############################################################################

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray


VALID_MOODS = ("neutral", "happy", "surprised", "thinking", "listening")


class EmotionMood:
    """情绪状态机：持有当前 mood 与过渡中的 (from, to, t0, duration)。"""

    def __init__(self, default_mood: str = "neutral", default_fade_ms: int = 250):
        if default_mood not in VALID_MOODS:
            default_mood = "neutral"
        self.current: str = default_mood
        self.default_fade_ms = int(default_fade_ms)
        # 过渡状态
        self._from_mood: Optional[str] = None
        self._to_mood: Optional[str] = None
        self._fade_start: float = 0.0
        self._fade_dur_s: float = 0.0

    # ---------------- 状态切换 ----------------
    def set_mood(self, name: str) -> None:
        """硬切：立即生效，无过渡。未知 mood → neutral。"""
        if name not in VALID_MOODS:
            name = "neutral"
        self.current = name
        self._from_mood = None
        self._to_mood = None

    def fade_to(self, name: str, ms: Optional[int] = None, now: Optional[float] = None) -> None:
        """软过渡：ms 内从 current 淡入到 name。"""
        if name not in VALID_MOODS:
            name = "neutral"
        if name == self.current:
            return
        if now is None:
            now = time.monotonic()
        dur_ms = int(ms) if ms is not None else self.default_fade_ms
        self._from_mood = self.current
        self._to_mood = name
        self._fade_start = now
        self._fade_dur_s = max(0.001, dur_ms / 1000.0)
        # current 目标态立即更新，供其它层查询
        self.current = name

    def current_alpha(self, now: Optional[float] = None) -> tuple[Optional[str], Optional[str], float]:
        """返回 (from_mood, to_mood, alpha∈[0,1])；无过渡时 (None, current, 1.0)。"""
        if self._from_mood is None or self._to_mood is None:
            return (None, self.current, 1.0)
        if now is None:
            now = time.monotonic()
        t = (now - self._fade_start) / self._fade_dur_s
        if t >= 1.0:
            # 过渡完成
            done_to = self._to_mood
            self._from_mood = None
            self._to_mood = None
            return (None, done_to, 1.0)
        return (self._from_mood, self._to_mood, max(0.0, min(1.0, t)))

    # ---------------- 合成 pass（P1 才生效；P0 无 expression 图集 → pass-through） ----------------
    def apply(self, frame: NDArray[np.uint8], ctx: dict) -> NDArray[np.uint8]:
        """约定 ctx 需含：'now'、可选 'expression_frames'（dict[str, NDArray]，与 frame 同尺寸）。"""
        from_mood, to_mood, alpha = self.current_alpha(ctx.get("now"))
        ctx["mood"] = to_mood
        ctx["mood_alpha"] = alpha

        expr_map = ctx.get("expression_frames")
        if expr_map is None:
            return frame

        import cv2
        to_img = expr_map.get(to_mood)
        if to_img is None:
            return frame
        # 无过渡或 from 缺失：直接贴 to（等价整帧图集切换）
        if from_mood is None or alpha >= 0.999:
            return to_img.copy()
        from_img = expr_map.get(from_mood)
        if from_img is None:
            return to_img.copy()
        # 淡入淡出
        return cv2.addWeighted(from_img, 1.0 - alpha, to_img, alpha, 0)
