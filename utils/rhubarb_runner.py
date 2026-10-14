###############################################################################
#  Rhubarb CLI 封装 — 离线把 TTS wav 转成 viseme timeline JSON
#
#  Rhubarb Lip Sync (https://github.com/DanielSWolf/rhubarb-lip-sync) 是一款
#  开源音素/viseme 时间戳提取工具，输出 9 档 Hanna-Barbera viseme。
#
#  用法：
#      python utils/rhubarb_runner.py --wav path/to/tts.wav --out path/to/timeline.json
#
#  P0：仅接口，不影响主流程；未安装 rhubarb 时直接返回 skipped。
#  P1：TTS 输出 wav 后异步跑一次，写入 data/avatars/<id>/viseme_timelines/<uttid>.json；
#      主循环通过 VisemeTimeline.load(path) 消费。
###############################################################################

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional


# Rhubarb 输出（-f json）示例：
#   [{"startTime": "0:00:00.000", "endTime": "0:00:00.080", "value": "A"}, ...]
# 我们统一到 VisemeTimeline 期望的格式（start/end/value 保持不变即可）。


def find_rhubarb(bin_path: Optional[str] = None) -> Optional[str]:
    """定位 rhubarb 可执行文件：显式路径 > PATH > 常见安装目录。找不到返回 None。"""
    if bin_path and os.path.isfile(bin_path) and os.access(bin_path, os.X_OK):
        return bin_path
    which = shutil.which("rhubarb")
    if which:
        return which
    for p in (
        "/opt/homebrew/bin/rhubarb",
        "/usr/local/bin/rhubarb",
        "/usr/bin/rhubarb",
    ):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def run_rhubarb(
    wav_path: str,
    out_json: str,
    *,
    bin_path: Optional[str] = None,
    machine_lib: Optional[str] = None,
    timeout_s: int = 60,
) -> tuple[bool, str]:
    """调用 Rhubarb CLI 生成 JSON timeline。

    返回 (ok, message)。ok=False 时 out_json 可能未生成；调用方应回退 RMS 路径。
    """
    exe = find_rhubarb(bin_path)
    if exe is None:
        return (False, "rhubarb not installed; set --rhubarb_bin or place on PATH")
    if not os.path.isfile(wav_path):
        return (False, f"wav 不存在: {wav_path}")

    # Rhubarb 要求 16-bit PCM mono WAV；TTS 常见输出可能是 44.1k/浮点，做一次规范化
    normalized_wav = _ensure_pcm16_mono(wav_path)
    if normalized_wav is None:
        return (False, "无法规范化 wav 到 16-bit PCM mono")

    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)

    cmd = [exe, "-f", "json", "-o", out_json, normalized_wav]
    if machine_lib:
        cmd += ["--model", machine_lib]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return (False, f"rhubarb timeout ({timeout_s}s)")
    except FileNotFoundError as e:
        return (False, f"rhubarb 执行失败: {e}")

    if proc.returncode != 0:
        return (False, f"rhubarb rc={proc.returncode}: {proc.stderr.strip()[:400]}")

    # 简单校验产物可解析
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return (False, "rhubarb 输出非数组")
    except Exception as e:  # noqa: BLE001
        return (False, f"rhubarb 输出 JSON 解析失败: {e}")

    return (True, f"timeline 已写入 {out_json}")


def _ensure_pcm16_mono(wav_path: str) -> Optional[str]:
    """确保 wav 为 16-bit PCM mono；不符则用 ffmpeg 转到临时文件。返回可用路径或 None。"""
    import wave
    try:
        with wave.open(wav_path, "rb") as w:
            if w.getnchannels() == 1 and w.getsampwidth() == 2:
                return wav_path
    except Exception:
        pass  # 交给 ffmpeg 处理

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", wav_path, "-ac", "1", "-ar", "16000",
             "-sample_fmt", "s16", tmp],
            capture_output=True, timeout=30, check=True,
        )
        return tmp
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None


# ---------------- CLI ----------------
def _main() -> int:
    ap = argparse.ArgumentParser(description="Rhubarb CLI wrapper for LiveTalking mouthdict")
    ap.add_argument("--wav", required=True, help="TTS 输出 wav 路径")
    ap.add_argument("--out", required=True, help="viseme timeline JSON 输出路径")
    ap.add_argument("--rhubarb-bin", default="", help="rhubarb 可执行文件绝对路径（可选）")
    ap.add_argument("--model", default="", help="rhubarb machine learning model 目录（可选）")
    args = ap.parse_args()

    ok, msg = run_rhubarb(
        args.wav, args.out,
        bin_path=args.rhubarb_bin or None,
        machine_lib=args.model or None,
    )
    print(("[OK] " if ok else "[FAIL] ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
