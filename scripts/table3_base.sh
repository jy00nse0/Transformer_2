#!/bin/bash
#bash scripts/table3_base.sh | tee logs/table3_base$(date +%Y%m%d_%H%M).log
# 실험 명칭 및 경로 설정
EXP_NAME="table3_base"
VOCAB_DIR="artifacts_pretrained"
CHECKPOINT_DIR="checkpoints/${EXP_NAME}"
LOG_FILE="logs/${EXP_NAME}_$(date +%Y%m%d_%H%M).log"

mkdir -p ${CHECKPOINT_DIR}
mkdir -p logs

echo "================================================================================"
echo "🚀 Starting Experiment: ${EXP_NAME} (Transformer Base)"
echo "   Save directory: ${CHECKPOINT_DIR}"
echo "================================================================================"

# 1. Training (Paper: 100,000 steps)
# H100의 80GB VRAM을 고려하여 max_tokens를 60,000으로 상향 조정했습니다.
python demo_wmt14_pretrained.py \
    --load_dir ${VOCAB_DIR} \
    --save_dir ${VOCAB_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --max_steps 100000 \
    --max_tokens 25000 \
    --num_workers 4 \
    --gradient_checkpointing \
    --log_every 100 \
    --checkpoint_every 10000 2>&1 | tee ${LOG_FILE}

# 2. Inference & Evaluation (Paper: Average last 5 checkpoints)
echo -e "\n\n"
echo "================================================================================"
echo "📊 Starting Inference & BLEU Evaluation for ${EXP_NAME}"
echo "================================================================================"

python inference.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --vocab_dir ${VOCAB_DIR} \
    --avg_checkpoints 5 \
    --decode_method beam \
    --beam_size 4 \
    --length_penalty 0.6 \
    --output_file "results/${EXP_NAME}_translations.txt" 2>&1 | tee -a ${LOG_FILE}

echo "✅ Experiment ${EXP_NAME} Completed!"
