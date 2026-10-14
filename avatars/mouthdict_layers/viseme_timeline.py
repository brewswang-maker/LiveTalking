###############################################################################
#  VisemeTimeline — Rhubarb JSON 加载 + 与 RMS 融合（P0：接口预留，未接入 = 返回 None）
#
#  Rhubarb CLI 输出示例（简化）：
#    [{"start": "0:00:00.000", "end": "0:00:00.080", "value": "A"}, ...]
#  我们把它归一化为：[{"t_ms": int, "level": int}, ...]（level 由 viseme→档位映射表决定）
###############################################################################

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from utils.logger import logger


# Hanna-Barbera / Rhubarb 常见 9 viseme → 档位建议（默认按张开度分档）
# 未知 viseme → 保持当前档位不变
DEFAULT_VISEME_TO_LEVEL: dict[str, int] = {
    # Rhubarb: A C D E F G H  + 自定义 I/O
    "A": 3,   # AA / "cat"  大张
    "B": 1,   # PP / MBP    闭合
    "C": 2,   # FF / EC     半开
    "D": 2,   # TTH / DH    半开
    "E": 1,   # SS / ZH     微张
    "F": 3,   # K / G       大张
    "G": 4,   # OH / OW     圆唇大张
    "H": 1,   # Q           微张（oo 起始）
    "I": 0,   # YY / EE     扁平（微笑）
    "O": 4,   # 圆唇 O
    "U": 3,   # 圆唇 U
}


def _parse_ts(s: str) -> int:
    """Rhubarb 时间戳 "H:MM:SS.mmm" → ms。"""
    try:
        h, m, sec = s.split(":")
        sec_i, ms = (sec.split(".") + ["0"])[:2]
        return (int(h) * 3600 + int(m) * 60 + int(sec_i)) * 1000 + int(ms[:3])
    except Exception:
        return 0


# Rhubarb viseme 值 → 嘴形家族（P1 文本驱动路线用家族，level 仍可由 vmap 提供）
VISEME_TO_FAMILY: dict[str, str] = {
    "A": "open", "F": "open", "D": "open",
    "B": "closed", "E": "closed",
    "C": "wide", "I": "wide",
    "G": "round", "H": "round", "O": "round", "U": "round",
}


class VisemeTimeline:
    """按 frame time_ms 查询目标 viseme；未命中或超时返回 None（走 RMS 兜底）。

    P1 扩展：内部段统一为 (start_ms, end_ms, family, level|None)。
    - 文本路线 set_from_text()：family 驱动嘴形形状，张开度仍由 RMS 定
    - 旧 Rhubarb JSON 路线 load()：value → family + level 双写，向后兼容 lookup()
    """

    def __init__(
        self,
        viseme_map: Optional[dict[str, int]] = None,
        frame_ms: int = 40,
        tolerance_ms: int = 80,
    ):
        self.vmap = dict(viseme_map or DEFAULT_VISEME_TO_LEVEL)
        self.frame_ms = int(frame_ms)
        self.tolerance = int(tolerance_ms)
        self._entries: List[tuple[int, int, Optional[str], Optional[int]]] = []
        self._loaded_from: Optional[str] = None

    # ---------------- 加载 ----------------
    def set_from_text(self, text: str, duration_ms: int) -> bool:
        """P1 文本路线：中文文本 + 音频时长 → 家族时间线。失败返回 False（回退 RMS）。"""
        from avatars.mouthdict_layers.text_viseme import build_timeline
        segs = build_timeline(text, duration_ms)
        if not segs:
            return False
        self._entries = [(s, e, fam, None) for s, e, fam in segs]
        self._loaded_from = f"text:{text[:12]}"
        return True

    def load(self, json_path: str) -> bool:
        """从 Rhubarb timeline JSON 加载；文件不存在/解析失败 → False，走 RMS。"""
        if not json_path or not os.path.exists(json_path):
            return False
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"VisemeTimeline 加载失败 {json_path}: {e}")
            return False

        entries: List[tuple[int, int, Optional[str], Optional[int]]] = []
        for item in raw:
            start = _parse_ts(item.get("start", "0:00:00.000"))
            end = _parse_ts(item.get("end", "0:00:00.000"))
            value = item.get("value", "A")
            level = self.vmap.get(value)
            family = VISEME_TO_FAMILY.get(value)
            if level is None and family is None:
                continue
            entries.append((start, end, family, level))
        self._entries = entries
        self._loaded_from = json_path
        logger.info(f"VisemeTimeline 载入 {len(entries)} 段 ← {json_path}")
        return True

    def reset(self) -> None:
        self._entries = []
        self._loaded_from = None

    # ---------------- 查询 ----------------
    def lookup(self, t_ms: int) -> Optional[int]:
        """按帧时间（毫秒）查询目标 level（旧契约）；无 level 信息时由 family 推导。"""
        hit = self._find(t_ms)
        if hit is None:
            return None
        _, _, family, level = hit
        if level is not None:
            return level
        return 0 if family == "closed" else 3   # 家族→粗略张开度兜底

    def lookup_family(self, t_ms: int) -> Optional[str]:
        """P1：按帧时间查询嘴形家族 open/round/wide/closed；未命中返回 None。"""
        hit = self._find(t_ms)
        return hit[2] if hit and hit[2] else None

    def _find(self, t_ms: int):
        if not self._entries:
            return None
        # 线性扫描（时间轴通常 <200 段，帧预算内可接受）；可优化为 bisect
        for start, end, family, level in self._entries:
            if start - self.tolerance <= t_ms <= end + self.tolerance:
                return start, end, family, level
            if start > t_ms + self.tolerance:
                break
        return None

    @property
    def loaded(self) -> bool:
        return bool(self._entries)

    @property
    def source_path(self) -> Optional[str]:
        return self._loaded_from
