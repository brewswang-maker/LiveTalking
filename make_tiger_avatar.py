#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把卡通小老虎素材打包成 LiveTalking wav2lip 可用的 avatar + 待机动作。

设计（对应本项目机制）：
  - 说话素材(talking)  -> data/avatars/<avatar_id>/  : full_imgs + face_imgs + coords.pkl
        说话时 wav2lip 只重绘 face 裁剪框的"下半部分"，所以 bbox 要让嘴巴落在框的下半区。
        卡通脸自动检测会失败，这里【跳过人脸检测】，直接用手动 bbox 裁切并写 coords.pkl。
  - 待机素材(idle)     -> data/avatars/<avatar_id>_idle/ : 25fps 帧序列
        注册为 audiotype=1 的自定义状态；静音时音频帧 type=1，渲染层会自动循环播放该视频。
        （见 base_asr.get_audio_frame / base_avatar.process_frames）

依赖：ffmpeg / ffprobe 在 PATH 中；opencv-python。
纯 CPU，可在本机跑；真正的推理放到 T4 服务器。

用法示例：
  python make_tiger_avatar.py \
      --talking "/Users/mac1234/项目/huadunold.mp4" \
      --idle    "/Users/mac1234/项目/数字人/华盾吉祥物/数字人/huaxiaodunbig.mp4" \
      --avatar_id huadun \
      --bbox 210 470 560 770
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys

import cv2


def run(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_frames(video, out_dir, out_w, out_h, fps):
    """抽帧并重采样到 fps、缩放到 out_w x out_h（保持 9:16，轻微裁剪避免拉伸）。"""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # scale 覆盖后再居中裁剪到目标比例，最后精确 resize 到 out_w:out_h
    vf = (
        f"fps={fps},"
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h}"
    )
    run([
        "ffmpeg", "-y", "-v", "error", "-i", video,
        "-vf", vf,
        os.path.join(out_dir, "%08d.png"),
    ])


def list_png(d):
    return sorted(
        [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".png")],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )


def build_face_and_coords(full_dir, face_dir, bbox, img_size):
    """按 bbox 裁出人脸区域 -> resize img_size -> face_imgs；coords.pkl 存 (y1,y2,x1,x2)。"""
    if os.path.isdir(face_dir):
        shutil.rmtree(face_dir)
    os.makedirs(face_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox
    coords = []
    for p in list_png(full_dir):
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(W, x2), min(H, y2)
        crop = img[cy1:cy2, cx1:cx2]
        crop = cv2.resize(crop, (img_size, img_size))
        cv2.imwrite(os.path.join(face_dir, os.path.basename(p)), crop)
        coords.append((int(cy1), int(cy2), int(cx1), int(cx2)))
    with open(os.path.join(os.path.dirname(full_dir), "coords.pkl"), "wb") as f:
        pickle.dump(coords, f)
    return coords


def write_debug_overlay(full_dir, bbox, out_path):
    """把 bbox 画在第一帧上，供人工核对嘴是否落在框的下半区。"""
    frames = list_png(full_dir)
    if not frames:
        return
    img = cv2.imread(frames[0])
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    midy = (y1 + y2) // 2
    cv2.line(img, (x1, midy), (x2, midy), (0, 255, 0), 2)  # 中线：嘴应在此线下方
    cv2.putText(img, "red=bbox  green=mid(voice zone below)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--talking", required=True, help="说话素材视频路径(基础avatar)")
    ap.add_argument("--idle", required=True, help="待机动作视频路径(audiotype=1)")
    ap.add_argument("--avatar_id", default="huadun")
    ap.add_argument("--bbox", type=int, nargs=4, default=[210, 470, 560, 770],
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="在输出分辨率下的嘴部裁剪框，需保证嘴巴落在框的下半区")
    ap.add_argument("--out_w", type=int, default=720)
    ap.add_argument("--out_h", type=int, default=1280)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--img_size", type=int, default=256, help="wav2lip256 用 256")
    ap.add_argument("--save_audio", action="store_true",
                    help="保留说话素材原音(一般不需要, 由TTS驱动)")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"未找到 {tool}，请先安装 ffmpeg")

    base = os.path.join("data", "avatars", args.avatar_id)
    idle_dir = os.path.join("data", "avatars", f"{args.avatar_id}_idle")
    full_dir = os.path.join(base, "full_imgs")
    face_dir = os.path.join(base, "face_imgs")

    print("== 1/4 抽取说话素材帧 (25fps, 统一分辨率) ==")
    extract_frames(args.talking, full_dir, args.out_w, args.out_h, args.fps)

    print("== 2/4 按手动 bbox 生成 face_imgs + coords.pkl ==")
    coords = build_face_and_coords(full_dir, face_dir, args.bbox, args.img_size)
    print(f"   共 {len(coords)} 帧，bbox={args.bbox}")

    print("== 3/4 抽取待机素材帧 (audiotype=1) ==")
    extract_frames(args.idle, idle_dir, args.out_w, args.out_h, args.fps)

    print("== 4/4 写 custom.json (待机自动播放) ==")
    custom = [{"audiotype": 1, "imgpath": idle_dir}]
    with open("custom.json", "w", encoding="utf-8") as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)
    print("   -> custom.json")

    # 标定预览图
    dbg = os.path.join("/tmp", f"{args.avatar_id}_bbox_debug.png")
    write_debug_overlay(full_dir, args.bbox, dbg)
    print(f"   bbox 预览: {dbg}")

    print("\n完成。T4 服务器启动命令：")
    print(f"  python app.py --transport webrtc --model wav2lip "
          f"--avatar_id {args.avatar_id} --modelres 256 --batch_size 4 "
          f"--customvideo_config custom.json")


if __name__ == "__main__":
    main()
