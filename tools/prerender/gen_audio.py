#!/usr/bin/env python3
"""P2 步骤①：从 prerender_manifest.json 批量生成 TTS 音频（本地即可跑）。

产出: data/prerender/audio/<id>.mp3 + <id>.wav (16k mono, LongCat 直接可用)
依赖: pip install edge-tts；ffmpeg 在 PATH（brew install ffmpeg）
"""
import asyncio
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
MANIFEST = os.path.join(ROOT, 'data', 'prerender_manifest.json')


async def synth_one(text: str, voice: str, out_mp3: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_mp3)


def main():
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print("缺依赖: pip install edge-tts", file=sys.stderr)
        sys.exit(1)

    mf = json.load(open(MANIFEST, encoding='utf-8'))
    voice = mf.get('voice', 'zh-CN-YunxiaNeural')
    out_dir = os.path.join(ROOT, mf.get('output_dir', 'data/prerender'), 'audio')
    os.makedirs(out_dir, exist_ok=True)

    async def run_all():
        for item in mf['items']:
            mp3 = os.path.join(out_dir, f"{item['id']}.mp3")
            wav = os.path.join(out_dir, f"{item['id']}.wav")
            if os.path.isfile(wav):
                print(f"跳过 {item['id']}（已存在）")
                continue
            await synth_one(item['text'], voice, mp3)
            subprocess.run(
                ['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3,
                 '-ar', '16000', '-ac', '1', wav], check=True)
            dur = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                 '-of', 'csv=p=0', wav], capture_output=True, text=True).stdout.strip()
            print(f"✓ {item['id']}  {float(dur):.1f}s")

    asyncio.run(run_all())
    print(f"\n音频产出目录: {out_dir}")
    print("下一步（P2 步骤②）：连同参考图上传云 GPU，跑 LongCat-1.5 批量出片，"
          "命令模板见 tools/prerender/README.md")


if __name__ == '__main__':
    main()
