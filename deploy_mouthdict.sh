#!/usr/bin/env bash
# =============================================================================
# mouthdict（含 P1 文本 viseme + 内容遮罩）一键部署到 T4 服务器
# 用法：
#   TARGET=<用户>@<IP> bash deploy_mouthdict.sh
#   例：TARGET=root@62.234.29.115 SSH_PORT=5222 SSH_KEY=~/ssh/id_rsa bash deploy_mouthdict.sh
# 可覆盖变量：REMOTE_DIR(默认/gpt/LiveTalking) CONDA_BASE(默认/gpt/anaconda3)
#             ENV_NAME(默认livetalking) SSH_PORT(默认22) SSH_KEY(默认空=用系统密钥)
# =============================================================================
set -euo pipefail

TARGET="${TARGET:-}"
REMOTE_DIR="${REMOTE_DIR:-/gpt/LiveTalking}"
CONDA_BASE="${CONDA_BASE:-/gpt/anaconda3}"
ENV_NAME="${ENV_NAME:-livetalking}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-}"

if [[ -z "$TARGET" ]]; then
    echo "用法: TARGET=<用户>@<IP> [SSH_PORT=5222] [SSH_KEY=~/.ssh/id_rsa] bash deploy_mouthdict.sh"
    exit 1
fi

cd "$(dirname "$0")"

if [[ -n "$SSH_KEY" ]]; then
    RSH="ssh -p $SSH_PORT -i $SSH_KEY -o StrictHostKeyChecking=no"
else
    RSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=no"
fi
HOST="${TARGET#*@}"; USER="${TARGET%@*}"

echo "── 1/4 推送嘴部资产（14 帧口型表[阴影截断修复版] + manifest + 头部对齐偏移）"
rsync -avz -e "$RSH" data/avatars/huadun/mouth_frames/ \
    "$TARGET:$REMOTE_DIR/data/avatars/huadun/mouth_frames/"
rsync -avz -e "$RSH" data/avatars/huadun/mouth_align.pkl \
    "$TARGET:$REMOTE_DIR/data/avatars/huadun/"

echo "── 2/4 推送代码（mouthdict 主体+内容遮罩 + 四层子包 + TTS duration_ms + 配置）"
rsync -avz -e "$RSH" avatars/mouthdict_avatar.py "$TARGET:$REMOTE_DIR/avatars/"
rsync -avz -e "$RSH" avatars/mouthdict_layers/ "$TARGET:$REMOTE_DIR/avatars/mouthdict_layers/"
rsync -avz -e "$RSH" tts/edge.py tts/tencent.py "$TARGET:$REMOTE_DIR/tts/"
rsync -avz -e "$RSH" config.py requirements.txt app.py "$TARGET:$REMOTE_DIR/"

echo "── 3/4 安装 pypinyin（文本 viseme 依赖，缺失则嘴形只有张合没有形状）"
$RSH "$TARGET" "$CONDA_BASE/envs/$ENV_NAME/bin/pip install -q pypinyin" \
    && $RSH "$TARGET" "$CONDA_BASE/envs/$ENV_NAME/bin/python -c 'import pypinyin; print(\"pypinyin OK\", pypinyin.__version__)'"

echo "── 4/4 重启 mouthdict 服务（注意 --model mouthdict，非 wav2lip）"
$RSH "$TARGET" "cd $REMOTE_DIR \
    && pkill -f 'app.py --transport webrtc' || true; sleep 2 \
    && source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $ENV_NAME \
    && nohup python app.py --transport webrtc --model mouthdict --avatar_id huadun \
        --enable_blink 1 --enable_gaze 1 --enable_breath 1 --enable_emotion 1 \
        --batch_size 4 --customvideo_config custom.json --kb_path data/huadun_kb.json \
        --llm_provider zhipu --tts tencent --listenport 8011 \
        > livetalking.log 2>&1 & \
    sleep 8; grep -E 'mouth manifest|内容遮罩|mouthdict|Start webrtc|Error|Traceback' livetalking.log | tail -8"

cat <<EOF
─────────────────────────────────────────────
部署完成。验收要点：
  1. 日志出现 "mouth manifest 载入" + "mouth 内容遮罩生成"
  2. 说话时日志出现 "viseme timeline"（无此行=pypinyin 没生效，嘴形只有张合）
  3. 大张嘴时嘴下应干净无灰带（阴影截断修复）
  4. 眼睛/衣领全程无虚影（内容遮罩保证）
远程看日志: ssh -p $SSH_PORT -i $SSH_KEY $TARGET 'tail -f $REMOTE_DIR/livetalking.log'
EOF
