###############################################################################
#  LayerCompositor — mouthdict 四层合成管线（L0 肢体偏移 / L1 嘴形 / L2 眼神 / L3 表情）
#
#  管线顺序（与 Plan 一致）：
#      输入 body_frame → [L1 paste mouth] → [L2 paste eyes] → [L3 emotion overlay]
#      → [L0 breath dy-offset] → 输出。
#
#  breath 作为最外层，对已合成画面做整体平移，保证 body+mouth+eyes+emotion 同步移动。
#
#  每层由独立的回调注入；LayerCompositor 只负责按开关调度与 pass-through 契约。
#  scheduler 与 avatar 主类的耦合发生在装配阶段（MouthDictReal 把回调注册进来），
#  compositor 本身与具体实现解耦。
###############################################################################

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from utils.logger import logger


# 每层回调签名：(frame_in: [H,W,3] uint8, ctx: dict) -> [H,W,3] uint8
# ctx 至少包含 idx / audio_frames / speaking / mapper_smoothed / t_ms 等锚点。
LayerFn = Callable[[NDArray[np.uint8], dict], NDArray[np.uint8]]


class LayerCompositor:
    """四层合成管线，支持逐层开关 + 素材缺失时 pass-through。

    使用方式：
        comp = LayerCompositor()
        comp.register("breath", breath_scheduler.apply, enabled=opt.enable_breath)
        comp.register("mouth",  mouthdict_avatar._paste_mouth_layer, enabled=True)
        comp.register("eyes",   blink_and_gaze_apply, enabled=opt.enable_blink or opt.enable_gaze)
        comp.register("emotion",emotion_mood.apply,   enabled=opt.enable_emotion)

        out = comp.compose(body_frame.copy(), ctx)

    契约：任意层 disabled 或回调返回 None，视为 pass-through 返回入参帧。
    """

    # 顺序即叠加顺序（内 → 外：mouth → eyes → emotion → breath）
    # breath 放最后作为最外层，保证已合成的 body+mouth+eyes+emotion 整体平移。
    LAYER_ORDER = ("mouth", "eyes", "emotion", "breath")

    def __init__(self):
        self._layers: dict[str, tuple[LayerFn, bool]] = {}

    # ---------------- 注册 / 开关 ----------------
    def register(self, name: str, fn: Optional[LayerFn], *, enabled: bool = True) -> None:
        if name not in self.LAYER_ORDER:
            raise ValueError(f"未知层名 {name!r}，合法值: {self.LAYER_ORDER}")
        if fn is None:
            # 显式允许 None → pass-through
            self._layers[name] = (lambda frame, ctx: frame, enabled)
            return
        self._layers[name] = (fn, enabled)

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in self._layers:
            return
        fn, _ = self._layers[name]
        self._layers[name] = (fn, enabled)

    # ---------------- 合成 ----------------
    def compose(self, frame: NDArray[np.uint8], ctx: dict) -> NDArray[np.uint8]:
        """按 LAYER_ORDER 依次执行启用的层；任何异常都降级为 pass-through，不阻断主循环。"""
        out = frame
        for name in self.LAYER_ORDER:
            entry = self._layers.get(name)
            if entry is None:
                continue
            fn, enabled = entry
            if not enabled:
                continue
            try:
                nxt = fn(out, ctx)
            except Exception as e:  # noqa: BLE001 - 主循环保护
                logger.warning(f"LayerCompositor[{name}] error: {e}, pass-through")
                continue
            if nxt is None:
                # 显式 pass-through
                continue
            out = nxt
        return out

    # ---------------- 便捷查询 ----------------
    def is_active(self, name: str) -> bool:
        entry = self._layers.get(name)
        return bool(entry and entry[1])

    def active_layers(self) -> list[str]:
        return [n for n in self.LAYER_ORDER if self.is_active(n)]
