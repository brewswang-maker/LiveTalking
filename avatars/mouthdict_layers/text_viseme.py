###############################################################################
#  text_viseme — 文本 → 拼音 → viseme 嘴形家族时间线（纯 CPU 查表，零依赖推理）
#
#  思路（Hanna-Barbera 六嘴型体系的中文简化版）：
#    中文按"一字一音节"均匀切分 TTS 音频时长，每个音节取其拼音韵母的
#    主导口型家族：open（大口 A）/ round（圆唇 O、U）/ wide（咧唇 EE/I）/
#    closed（闭唇 MB 及零元音 zi/ci/si 等）。
#    张开度（level 0~7）仍由全局 RMS 包络决定 —— 文本定"形状"，音量定"大小"。
#
#  输出段格式：(start_ms, end_ms, family)，family ∈ {open, round, wide, closed}
#  降级：pypinyin 不可用（服务器未装）→ 返回 None，上层回退纯 RMS（旧行为）。
###############################################################################

from __future__ import annotations

import re
from typing import Optional

try:
    from pypinyin import pinyin, Style
    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False

from utils.logger import logger

FAMILIES = ("open", "round", "wide", "closed")

# 按"最长后缀优先"匹配韵母 → 家族（如 iang 命中 ang 而非 ing）
_FINAL_RULES: list[tuple[str, str]] = [
    # 大口型：a/ai/ao/an/ang 及各类鼻音
    ("a", "open"), ("ai", "open"), ("ao", "open"), ("an", "open"), ("ang", "open"),
    ("o", "open"), ("e", "open"), ("ei", "open"), ("en", "open"), ("eng", "open"),
    ("er", "open"),
    # 咧唇扁平：i 系（ing/eng 系元音 i/e 主导，非鼻音韵尾）
    ("i", "wide"), ("ia", "wide"), ("ie", "wide"), ("iao", "wide"), ("iou", "wide"),
    ("ian", "wide"), ("in", "wide"), ("iang", "wide"), ("ing", "wide"),
    # 圆唇：u/ü/ou 系（iu=iou 收尾 u、ue/üe 主元音 ü；uan/uang/ve 主元音 a → open）
    ("u", "round"), ("ua", "round"), ("uo", "round"), ("uai", "round"),
    ("ong", "round"),
    ("ou", "round"), ("iu", "round"),
    ("v", "round"), ("vn", "round"),
    ("ü", "round"), ("üe", "round"), ("ün", "round"),
    ("ue", "round"),
    # u/ü 介音 + a/e 主元音：开口度由主元音主导
    ("uan", "open"), ("uang", "open"), ("ve", "open"), ("van", "open"),
]
# 长规则在前，避免 "iang" 被 "ang" 抢先或 "iu" 被 "i" 抢先
_FINAL_RULES.sort(key=lambda kv: -len(kv[0]))
_FINAL_MAP: dict[str, str] = dict(_FINAL_RULES)

# 零声母/成音节的字直接查表（pypinyin 对独立字返回完整音节）
_SPECIAL_SYLLABLE = {
    "m": "closed", "n": "closed", "ng": "closed",     # 嗯/唔
    "mm": "closed", "nen": "closed",
}
# 零元音音节（zh/ch/sh/r/z/c/s + i 行、y/w 起头的空读音节）→ 闭唇
_ZERO_VOWEL_PREFIX = ("zyz",)  # 占位，实际用音节集合判断

# 英文元音组 → 家族（混排英文/拼音字母的粗粒度回退）
_EN_DIGRAPH = {"oo": "round", "ou": "round", "ew": "round", "ui": "round",
               "ee": "wide", "ea": "wide", "ai": "open", "ay": "open",
               "oa": "open", "ow": "round", "au": "open", "aw": "open"}
_EN_MONO = {"a": "open", "e": "wide", "i": "wide", "o": "open", "u": "round"}

_CJK_RE = re.compile(r'[\u4e00-\u9fff]')
_VOWELS = set("aeiou")


def available() -> bool:
    return _HAS_PYPINYIN


def char_family(ch: str) -> Optional[str]:
    """单字 → 嘴形家族；无法判定返回 None。"""
    if not _HAS_PYPINYIN:
        return None
    if _CJK_RE.match(ch):
        py = pinyin(ch, style=Style.INITIALS, strict=False)[0][0]
        fin = pinyin(ch, style=Style.FINALS, strict=False)[0][0]
        # 零元音音节：zi/ci/si/zhi/chi/shi/ri（finals 只剩 i）
        if py + fin in ("zi", "ci", "si", "zhi", "chi", "shi", "ri"):
            return "closed"
        if fin in _SPECIAL_SYLLABLE:
            return _SPECIAL_SYLLABLE[fin]
        fam = _FINAL_MAP.get(fin)
        if fam is None and fin:
            fam = _FINAL_MAP.get(fin[-1]) or _FINAL_MAP.get(fin[0])
        return fam or "open"
    c = ch.lower()
    if c in _VOWELS:
        return _EN_MONO[c]
    return None


def _units_from_text(text: str) -> list[tuple[str, float]]:
    """文本 → [(family, weight)] 发音单元序列。

    - 汉字：1.0 权重一个音节
    - 字母：连续元音合并为一个单元（0.8），辅音并入后续单元
    - 标点/空白：0.35 权重的闭唇停顿单元
    - 音节前附 0.18 的闭唇辅音起始（声母感）
    """
    units: list[tuple[str, float]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _CJK_RE.match(ch):
            fam = char_family(ch)
            if fam:
                if fam != "closed":
                    units.append(("closed", 0.18))   # 声母闭合起始
                units.append((fam, 1.0))
            i += 1
        elif ch.isalpha():
            # 收集连续字母串
            j = i
            while j < n and text[j].isalpha() and not _CJK_RE.match(text[j]):
                j += 1
            word = text[i:j].lower()
            k = 0
            while k < len(word):
                if word[k] in _VOWELS:
                    grp = word[k]
                    if k + 1 < len(word) and word[k + 1] in _VOWELS:
                        grp = word[k:k + 2]
                    fam = _EN_DIGRAPH.get(grp) or _EN_MONO.get(grp[0], "open")
                    units.append((fam, 0.8))
                    k += len(grp)
                else:
                    k += 1
            i = j
        elif ch in "，。！？、；：,.!?;:…—~ \n\t":
            units.append(("closed", 0.35))
            i += 1
        else:
            i += 1
    return units


def build_timeline(text: str, duration_ms: int,
                   max_syllables: int = 60) -> Optional[list[tuple[int, int, str]]]:
    """文本 + 音频时长 → [(start_ms, end_ms, family)]；不可用/失败返回 None。"""
    if not _HAS_PYPINYIN or not text or duration_ms <= 200:
        return None
    units = _units_from_text(text)
    # 音节数保护：文本与音频时长严重失配（如 TTS 加戏/截断）→ 放弃文本路线
    n_syl = sum(w for _, w in units)
    if not units or n_syl < 1:
        return None
    est_ms_per_unit = duration_ms / n_syl
    if est_ms_per_unit < 60 or est_ms_per_unit > 1200:
        logger.debug(f"text_viseme: 时长/字数失配 ({duration_ms}ms / {n_syl}u)，回退 RMS")
        return None
    if len(units) > max_syllables * 3:
        return None
    total_w = sum(w for _, w in units)
    segs: list[tuple[int, int, str]] = []
    t = 0.0
    for fam, w in units:
        t_end = t + duration_ms * w / total_w
        segs.append((int(round(t)), int(round(t_end)), fam))
        t = t_end
    if segs:
        segs[-1] = (segs[-1][0], int(duration_ms), segs[-1][2])
    return segs
