# GPU 改造前的环境备份（回滚用）

## 改造前
torch      2.13.0+cpu
torchvision 0.28.0+cpu
CUDA 可用: False

## 回滚命令（如果 CUDA 版装坏了）
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

## 完整 freeze 见 pip_freeze_before_gpu.txt
