#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从说话视频里抽取"敬礼"片段，做成 LiveTalking 可触发的独立动作(custom state, audiotype=2)。

机制要点（见 base_avatar / base_asr）：
  - 自定义动作只在【静音帧】渲染，且触发依赖一条音频轨道(custom_audio_index)。
    因此为动作片段配一段【等长静音 wav】，时长 = 帧数/fps，保证视频与音频同步。
  - "先敬礼后说话"：客户端先 POST /set_audiotype {audiotype:2} 再 POST /human 文本。
    get_audio_frame 会优先把敬礼(静音)音频放完，语音排在其后播放 -> 敬礼结束后开口。

用法：
  python extract_salute_action.py \
      --video "/Users/mac1234/项目/huadunold.mp4" \
      --start 0.0 --end 2.5 --audiotype 2 --avatar_id huadun
"""
import argparse
import json
import os
import shutil
import subprocess
import sys


def run(cmd):
    print("[cmd]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def count_frames(d):
    return len([f for f in os.listdir(d) if f.lower().endswith(".png")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="含敬礼动作的源视频")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=2.5)
    ap.add_argument("--audiotype", type=int, default=2, help="动作编号(>1)")
    ap.add_argument("--avatar_id", default="huadun")
    ap.add_argument("--out_w", type=int, default=720)
    ap.add_argument("--out_h", type=int, default=1280)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--custom_json", default="custom.json")
    args = ap.parse_args()

    dur = args.end - args.start
    if dur <= 0:
        sys.exit("end 必须大于 start")

    frames_dir = os.path.join("data", "avatars", f"{args.avatar_id}_salute")
    wav_path = os.path.join("data", "avatars", f"{args.avatar_id}_salute.wav")

    # 1) 抽取敬礼帧序列（重采样到 fps、统一分辨率）
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    vf = (
        f"fps={args.fps},"
        f"scale={args.out_w}:{args.out_h}:force_original_aspect_ratio=increase,"
        f"crop={args.out_w}:{args.out_h}"
    )
    run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", str(args.start), "-to", str(args.end), "-i", args.video,
        "-vf", vf,
        os.path.join(frames_dir, "%08d.png"),
    ])
    n = count_frames(frames_dir)
    if n == 0:
        sys.exit("未抽到任何帧，检查 start/end 是否超出视频时长")

    # 2) 生成等长静音 wav（16kHz 单声道，时长 = n/fps），供触发与同步用
    actual_dur = n / args.fps
    run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
        "-t", f"{actual_dur:.3f}",
        "-acodec", "pcm_s16le", wav_path,
    ])
    print(f"敬礼片段：{n} 帧，时长 {actual_dur:.2f}s")

    # 3) 合并写入 custom.json（保留已有的待机 audiotype=1 等）
    custom = []
    if os.path.exists(args.custom_json):
        try:
            with open(args.custom_json, "r", encoding="utf-8") as f:
                custom = json.load(f)
        except Exception:
            custom = []
    entry = {"audiotype": args.audiotype, "imgpath": frames_dir, "audiopath": wav_path}
    custom = [c for c in custom if c.get("audiotype") != args.audiotype]  # 去重
    custom.append(entry)
    custom.sort(key=lambda c: c.get("audiotype", 0))
    with open(args.custom_json, "w", encoding="utf-8") as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)
    print(f"已更新 {args.custom_json}：新增 audiotype={args.audiotype} (敬礼)")

    print("\n完成。用法：")
    print("  1) 启动加 --customvideo_config custom.json")
    print(f"  2) 先敬礼：POST /set_audiotype {{\"sessionid\":\"..\",\"audiotype\":{args.audiotype}}}")
    print("  3) 再说话：POST /human {\"sessionid\":\"..\",\"type\":\"echo\",\"text\":\"你好\"}")
    print("     （语音会排在敬礼静音轨道之后播放 -> 敬礼结束即开口）")


if __name__ == "__main__":
    main()
