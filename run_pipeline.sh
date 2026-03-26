#!/bin/bash
set -e

#echo "=== 1. 单卡 A800: 使用 72B 量化模型生成微调数据 ==="
#CUDA_VISIBLE_DEVICES=0 python 1_data_generation.py

echo "=== 2. 单卡 A800: 使用 QLoRA 微调 30B 模型 ==="
CUDA_VISIBLE_DEVICES=0 python 2_finetune.py

echo "=== 3. 构建本地 RAG 知识库 ==="
python 3_build_vector_db.py

echo "=== 4. 启动 Web 服务 ==="
CUDA_VISIBLE_DEVICES=0 python 4_app.py