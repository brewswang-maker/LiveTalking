# P2：LongCat-1.5 离线预渲染高频回答（借云卡一次）

目标：把 12 条固定话术渲染成小老虎"真张嘴"的高质量视频资产，实时对话命中触发词时
直接播放原片（音视频合成流），质感超过任何 T4 实时方案。模型 MIT 可商用。

## 实测约束（决定为什么是离线而不是实时）

- LongCat-Video-Avatar 1.5：每 1 秒成片 ≈ 44 秒 GPU 计算（A800-40G，INT8+8步蒸馏）
- 稳定运行底线 40G 显存单卡；T4 16G 跑不动；多卡 context-parallel 有死锁报告
- 12 条 × ~8s ≈ 96s 成片 → 约 70-80 分钟单卡时间，租用 A800（ autodl/恒源云 ≈¥2-6/时）
  一次批量成本 ≈ ¥10 以内

## 步骤①（本地）：生成 TTS 音频

```bash
pip install edge-tts          # 一次性
python3 tools/prerender/gen_audio.py
# → data/prerender/audio/*.wav (16k mono)
```

## 步骤②（云 GPU）：批量出片

```bash
# 租用 40G+ 卡（A800/H20/4090-48G 均可），环境按官方 repo 部署：
git clone https://github.com/meituan-longcat/LongCat-Video
# 下载 LongCat-Video-Avatar-1.5 权重（HF/ModelScope），启用 INT8 + 8-step 蒸馏推理模式

# 上传：本仓库 data/prerender/audio/*.wav + 参考图
#   参考图 = data/avatars/huadun/full_imgs/00000087.png（正脸基准帧）
# 逐条推理模板（以 repo 的 avatar 推理脚本为准）：
python inference_avatar.py \
    --ref_image 00000087.png \
    --audio {id}.wav \
    --distill --int8 --steps 8 \
    --output {id}.mp4
```

验收清单（卡通崩坏检查项，逐条看）：
- [ ] 虎纹/描边无错位、无融化
- [ ] 眼睛不变形（LongCat 对动漫/动物官方支持，但需实测本形象）
- [ ] 身份一致：12 条视频与实时 mouthdict 形象并排看不出"两个人"
- [ ] 口型与音频同步（Whisper-large-v3 编码，中文应优于 wav2lip）
- [ ] 衣领/圆牌无伪影（扩散模型整帧生成，此项预期天然通过）

不合格条目：换 seed 重跑；连续 3 次崩坏 → 该条降级用 mouthdict 实时渲染兜底。

## 步骤③（回 T4）：接入实时系统

方案 A（推荐，改动小）：命中触发词 → 不走 LLM/TTS，把预渲染 mp4 当作
"自定义动作视频"播放（复用现有 customvideo + `/set_audiotype` 机制，
mp4 拆帧成 full_imgs 序列 + 音轨入 custom_audio）。

方案 B（体验更好，改动中）：在 server 路由层加 prerender 命中表
（manifest trigger 关键词），命中直接推 mp4 文件流经 ffmpeg 进 WebRTC 音视频轨道，
播完自动回实时 mouthdict 态。

manifest（触发词↔资产映射）：`data/prerender_manifest.json`

## 执行顺序建议

1. 先部署 P0+P1 mouthdict 上线验证（`TARGET=<用户>@<IP> bash deploy_mouthdict.sh`）
2. 实时链路稳定 1-2 天后，再约一个下午跑步骤②（云卡按小时计费，一次跑完 12 条）
3. 步骤③接入后，迎宾场景高频问题走原片、长尾问题走实时 mouthdict
