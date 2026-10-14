###############################################################################
#  配置解析 — CLI 参数 + YAML 配置
###############################################################################

import argparse
import json
import os

try:
    import yaml
    _has_yaml = True
except ImportError:
    _has_yaml = False


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def _yaml_to_args(yaml_cfg):
    """将 YAML 字典中的 key 转换为 argparse 兼容的 `--key` 形式。

    argparse 的 dest 默认规则：`--model` → `model`，`--push-url` → `push_url`。
    此函数同时支持两种 key 写法：
      - model / batch_size          → 直接透传
      - model-name / batch-size    → 转换为 model_name / batch_size
    """
    result = {}
    for k, v in yaml_cfg.items():
        dest = k.replace('-', '_')
        result[dest] = v
    return result


def parse_args():
    """解析命令行参数，支持 YAML 配置文件覆盖默认值。

    优先级：CLI 参数 > YAML 配置文件 > add_argument(default=...)
    """
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # ─── 配置文件 ──────────────────────────────────────────────────────
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='YAML 配置文件路径（设为空字符串可跳过）')

    # ─── 音频 ──────────────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=25, help="video fps, must be 25")
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    # ─── 画面 ──────────────────────────────────────────────────────────
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── 数字人模型 ────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/wav2lip/ultralight/mouthdict")
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')

    # ─── 查表驱动卡通嘴部 (model=mouthdict) ─────────────────────────
    parser.add_argument('--mouth_frames', type=str, default='',
                        help="mouth patch frames dir; empty = data/avatars/<id>/mouth_frames")
    parser.add_argument('--rms_floor', type=float, default=0.001,
                        help="RMS below this -> closed mouth")
    parser.add_argument('--rms_ceiling', type=float, default=0.2,
                        help="RMS above this -> widest mouth")
    parser.add_argument('--mouth_attack', type=float, default=0.6,
                        help="smoothing alpha when energy rises (open fast)")
    parser.add_argument('--mouth_release', type=float, default=0.2,
                        help="smoothing alpha when energy falls (close slow)")

    # ─── mouthdict 四层协同开关 (P0: 默认全关，与旧版 mouthdict 逐帧等价) ─────
    parser.add_argument('--enable_blink', type=int, default=0,
                        help="mouthdict: 启用眨眼调度器 (0/1); 无眼帧素材时自动 pass-through")
    parser.add_argument('--enable_gaze', type=int, default=0,
                        help="mouthdict: 启用视线追踪 (0/1); 无视线帧素材时自动 pass-through")
    parser.add_argument('--enable_breath', type=int, default=0,
                        help="mouthdict: 启用呼吸/肩动 dy 偏移 (0/1)")
    parser.add_argument('--enable_emotion', type=int, default=0,
                        help="mouthdict: 启用情绪整帧图集切换 (0/1); 无 expression 素材时 pass-through")
    parser.add_argument('--eye_frames_dir', type=str, default='',
                        help="眼帧目录; empty = data/avatars/<id>/eyes")
    parser.add_argument('--expression_dir', type=str, default='',
                        help="情绪图集目录; empty = data/avatars/<id>/expressions")
    parser.add_argument('--rhubarb_bin', type=str, default='',
                        help="Rhubarb CLI 路径; empty = 不启用 viseme timeline, 走 RMS")
    parser.add_argument('--viseme_timeline_dir', type=str, default='',
                        help="viseme timeline JSON 目录; empty = data/avatars/<id>/viseme_timelines")
    parser.add_argument('--enable_viseme_text', type=int, default=1,
                        help="mouthdict: 文本驱动嘴形家族 (0/1); 需 pypinyin + mouth_frames/manifest.json，缺失自动回退 RMS")
    parser.add_argument('--enable_occlusion', type=int, default=0,
                        help="mouthdict: 启用手遮挡跳帧 (0/1); 命中则跳过 mouth/eyes/emotion 层")

    # ─── 自定义动作和多形象 ────────────────────────────────────────────
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",
                        help="参考文件名或语音模型ID")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')

    # ─── LLM ──────────────────────────────────────────────────────────
    parser.add_argument('--llm_provider', type=str, default='dashscope',
                        help="llm provider: dashscope/orcarouter/zhipu/minimax")
    parser.add_argument('--llm_model', type=str, default='',
                        help="llm model override, empty = provider default (qwen-plus / orcarouter/auto)")
    parser.add_argument('--kb_path', type=str, default='data/huadun_kb.json',
                        help="公司知识库 JSON 路径（chat 模式注入 system prompt，如 data/huadun_kb.json）")

    # ─── 传输 ─────────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--stun', type=str, default='stun:stun.freeswitch.org:3478',
                        help="stun server url")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=5)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")

    # ─── 虚拟摄像头 ───────────────────────────────────────────────────
    parser.add_argument('--audio_output_device', type=int, default=None,
                        help="音频输出设备索引（None=系统默认，仅用于 --transport=virtualcam）。使用 python list_audio_devices.py 查看所有设备")

    # ─── 加载 YAML 配置文件 ────────────────────────────────────────────
    if _has_yaml:
        # 先用 parser 的已知参数做一次临时解析，只拿 --config 的值
        tmp_opt, _ = parser.parse_known_args()
        config_path = tmp_opt.config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_cfg = yaml.safe_load(f)
            if yaml_cfg and isinstance(yaml_cfg, dict):
                yaml_defaults = _yaml_to_args(yaml_cfg)
                parser.set_defaults(**yaml_defaults)
    else:
        print("[config] PyYAML 未安装，跳过 YAML 配置文件加载。"
              "安装: pip install pyyaml")

    # ─── 正式解析 CLI 参数 ─────────────────────────────────────────────
    opt = parser.parse_args()

    # ─── 后处理 ────────────────────────────────────────────────────────
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    return opt
