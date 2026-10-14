#!/usr/bin/env bash
# =============================================================================
# LiveTalking —— T4 (Linux + CUDA) 一键环境安装脚本
# 用法：
#   bash setup_t4.sh              # 完整安装（conda 环境 + torch + 依赖）
#   bash setup_t4.sh --check      # 只做环境自检，不安装
# 可覆盖变量（匹配你服务器上的驱动/CUDA 版本）：
#   ENV_NAME=livetalking  PYTHON=3.10  TORCH_CUDA=cu124  bash setup_t4.sh
#   # TORCH_CUDA 可选 cu118 / cu121 / cu124 / cu128；T4 老驱动优先 cu118/cu121
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-livetalking}"
PYTHON="${PYTHON:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"          # T4 常见可用；驱动很旧用 cu118
TORCH_VER="${TORCH_VER:-2.5.1}"            # 与 cu124 匹配的稳妥组合
MODEL_FILE="models/wav2lip.pth"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

echo "== 0. 基础检查 =="
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "!! 未检测到 nvidia-smi（确认在 GPU 服务器上运行）"
command -v ffmpeg >/dev/null && echo "ffmpeg: $(ffmpeg -version | head -1)" || echo "!! 缺 ffmpeg：sudo apt-get install -y ffmpeg"
[[ -f "$MODEL_FILE" ]] && echo "$MODEL_FILE: 已就位" || echo "!! 缺 $MODEL_FILE：需按 README 下载 wav2lip256.pth 并改名为 wav2lip.pth 放到 models/"

if [[ $CHECK_ONLY -eq 1 ]]; then
  echo "== 自检完成（--check 模式，不执行安装）=="
  exit 0
fi

echo "== 1. 创建 conda 环境 ${ENV_NAME} (python=${PYTHON}) =="
if ! command -v conda >/dev/null; then
  echo "!! 未找到 conda，请先安装 Miniconda：https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "$ENV_NAME" python="$PYTHON"
conda activate "$ENV_NAME"

echo "== 2. 安装 PyTorch (${TORCH_VER}, ${TORCH_CUDA}) =="
pip install --upgrade pip
pip install "torch==${TORCH_VER}" torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "== 3. 安装项目依赖 =="
pip install -r requirements.txt

echo "== 4. 校验 CUDA 可用性 =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM(GB):", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
else:
    print("!! CUDA 不可用：检查驱动版本是否支持所选 TORCH_CUDA（T4 需驱动 >= 对应 CUDA runtime 要求）")
PY

echo ""
echo "==================================================================="
echo " 环境就绪。下一步："
echo "  1) 确认 ${MODEL_FILE} 已放置（wav2lip256 权重）"
echo "  2) 启动小老虎数字人："
echo "     python app.py --transport webrtc --model wav2lip \\"
echo "         --avatar_id huadun --modelres 256 --batch_size 4 \\"
echo "         --customvideo_config custom.json --kb_path data/huadun_kb.json"
echo "     （echo 复读/问候零 key 可跑；问话交互走 chat 模式需在 .env 填 DASHSCOPE_API_KEY）"
echo "  3) 浏览器打开迎宾控制台：http://<服务器IP>:8010/huadun_reception.html"
echo "     注意放行端口：TCP 8010、UDP 1-65536"
echo "==================================================================="
