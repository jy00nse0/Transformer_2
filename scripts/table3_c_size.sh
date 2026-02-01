#!/bin/sh
# table3_c_size.sh

VOCAB_DIR="artifacts_pretrained"

# N=2, 4, 8 레이어 실험
for n in 2 4 8; do
    NAME="table3_c_n${n}"
    python demo_wmt14_pretrained.py --checkpoint_dir "checkpoints/${NAME}" --n_layers ${n} --max_steps 100000
    python inference.py --checkpoint_dir "checkpoints/${NAME}" --vocab_dir ${VOCAB_DIR} --output_file "results/${NAME}.txt"
done

# d_model & d_ff 변화 실험
# d_model=256 (d_ff=1024, dk=32)
python demo_wmt14_pretrained.py --checkpoint_dir "checkpoints/table3_c_dm256" --d_model 256 --ffn_hidden 1024 --kdim 32 --max_steps 100000
python inference.py --checkpoint_dir "checkpoints/table3_c_dm256" --vocab_dir ${VOCAB_DIR} --output_file "results/table3_c_dm256.txt"

# d_model=1024 (d_ff=4096, dk=128)
python demo_wmt14_pretrained.py --checkpoint_dir "checkpoints/table3_c_dm1024" --d_model 1024 --ffn_hidden 4096 --kdim 128 --max_steps 100000
python inference.py --checkpoint_dir "checkpoints/table3_c_dm1024" --vocab_dir ${VOCAB_DIR} --output_file "results/table3_c_dm1024.txt"