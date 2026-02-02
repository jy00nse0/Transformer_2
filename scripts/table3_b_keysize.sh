#!/bin/sh
# table3_b_keysize.sh
#bash scripts/table3_b_keysize.sh | tee logs/table3_b_$(date +%Y%m%d_%H%M).log

VOCAB_DIR="artifacts_pretrained"

# d_k=16 실험
python demo_wmt14_pretrained.py --load_dir ${VOCAB_DIR} --checkpoint_dir "checkpoints/table3_b_dk16" --kdim 16 --max_steps 100000 --gradient_checkpointing
python inference.py --checkpoint_dir "checkpoints/table3_b_dk16" --vocab_dir ${VOCAB_DIR} --output_file "results/table3_b_dk16.txt"

# d_k=32 실험
python demo_wmt14_pretrained.py --load_dir ${VOCAB_DIR} --checkpoint_dir "check points/table3_b_dk32" --kdim 32 --max_steps 100000 --gradient_checkpointing
python inference.py --checkpoint_dir "checkpoints/table3_b_dk32" --vocab_dir ${VOCAB_DIR} --output_file "results/table3_b_dk32.txt"
