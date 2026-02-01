# 토크나이저 저장/로드 기능 사용 가이드

## 주요 기능

### 1. 자동 저장
- BPE 토크나이저 (영어/독일어) 저장
- 어휘사전 저장
- 메타데이터 저장 (vocab 크기, 특수 토큰 인덱스 등)

### 2. 로드 및 스킵
- 저장된 토크나이저/어휘를 로드하여 재학습 방지
- 토크나이저 훈련 시간 절약 (8-12시간 → 0초)
- 바로 모델 학습 시작 가능

## 사용 방법

### 시나리오 1: 처음 실행 (토크나이저 학습 + 저장)

```bash
# 기본 실행 - artifacts/ 디렉토리에 자동 저장
python demo_wmt14_saveable.py

# 저장 경로 지정
python demo_wmt14_saveable.py --save_dir my_artifacts
```

**저장되는 파일:**
```
artifacts/
├── tokenizer_en.pkl      # 영어 BPE 토크나이저
├── tokenizer_de.pkl      # 독일어 BPE 토크나이저
├── vocab.pkl             # 어휘사전
└── metadata.json         # 메타데이터
```

**예상 소요 시간:**
- 데이터 로드: ~2분
- 토크나이저 훈련: 8-12시간 ⬅️ 한 번만!
- 어휘 구축: ~5분
- 저장: ~10초
- **총: 약 8-12시간**

### 시나리오 2: 저장된 결과물로 학습 (토크나이저 스킵)

```bash
# 저장된 결과물 로드하여 바로 모델 학습
python demo_wmt14_saveable.py --load_dir artifacts

# 다른 경로에서 로드
python demo_wmt14_saveable.py --load_dir my_artifacts
```

**예상 소요 시간:**
- 데이터 로드: ~2분
- 토크나이저/어휘 로드: **~5초** ⬅️ 엄청 빠름!
- Iterator 생성: ~30초
- 모델 학습: (에폭 수에 따라)
- **총: ~3분 + 학습 시간**

### 시나리오 3: 다양한 설정으로 실험

```bash
# 작은 vocab으로 빠른 테스트
python demo_wmt14_saveable.py --vocab_size 10000 --save_dir artifacts_10k

# 나중에 이 결과물로 여러 실험
python demo_wmt14_saveable.py --load_dir artifacts_10k --epochs 50
python demo_wmt14_saveable.py --load_dir artifacts_10k --epochs 100 --max_tokens 15000
```

## 명령행 옵션

### 필수 옵션

#### `--load_dir` (저장된 결과물 로드)
```bash
python demo_wmt14_saveable.py --load_dir artifacts
```
- 토크나이저 훈련 및 어휘 구축 스킵
- 지정된 디렉토리에서 로드
- 없으면 처음부터 학습

#### `--save_dir` (저장 경로 지정)
```bash
python demo_wmt14_saveable.py --save_dir my_output
```
- 기본값: `artifacts`
- 토크나이저와 어휘를 저장할 디렉토리

### 학습 옵션

#### `--epochs` (에폭 수)
```bash
python demo_wmt14_saveable.py --epochs 50
```
- 기본값: 100
- 학습 에폭 수

#### `--max_tokens` (배치당 토큰 수)
```bash
python demo_wmt14_saveable.py --max_tokens 15000
```
- 기본값: 25000 (논문 준수)
- GPU 메모리에 맞게 조정

#### `--vocab_size` (어휘 크기)
```bash
python demo_wmt14_saveable.py --vocab_size 10000
```
- 기본값: 37000 (논문 준수)
- 작을수록 빠르게 학습

#### `--checkpoint_dir` (체크포인트 저장 경로)
```bash
python demo_wmt14_saveable.py --checkpoint_dir my_checkpoints
```
- 기본값: `checkpoints`
- 모델 체크포인트 저장 디렉토리

## 실전 워크플로우

### Step 1: 토크나이저 한 번만 학습 (8-12시간)

```bash
# 논문 설정으로 토크나이저 학습
python demo_wmt14_saveable.py \
    --vocab_size 37000 \
    --save_dir artifacts_37k \
    --epochs 1
```

- `--epochs 1`: 1 에폭만 학습하고 중단 (Ctrl+C)
- 토크나이저와 어휘만 저장하면 됨

### Step 2: 다양한 실험 반복 (즉시 시작!)

```bash
# 실험 1: Base 모델 (논문 설정)
python demo_wmt14_saveable.py \
    --load_dir artifacts_37k \
    --epochs 100 \
    --max_tokens 25000 \
    --checkpoint_dir exp1_base

# 실험 2: 작은 배치
python demo_wmt14_saveable.py \
    --load_dir artifacts_37k \
    --epochs 100 \
    --max_tokens 15000 \
    --checkpoint_dir exp2_small_batch

# 실험 3: 긴 학습
python demo_wmt14_saveable.py \
    --load_dir artifacts_37k \
    --epochs 200 \
    --checkpoint_dir exp3_long
```

모든 실험이 **즉시 시작**됩니다!

## 출력 예시

### 처음 실행 (저장)

```
================================================================================
                    WMT14 Transformer Training
================================================================================

Configuration:
  Load directory: None (train from scratch)
  Save directory: artifacts
  Epochs: 100
  Max tokens per batch: 25,000
  Vocabulary size: 37,000
  Checkpoint directory: checkpoints

================================================================================
1. 데이터셋 로드
================================================================================

Loading WMT14 dataset...
...

================================================================================
2. BPE 토크나이저 훈련
================================================================================

Training English BPE Tokenizer
...
[Step 4/4] Performing BPE merges...
  Merges: 36,687/36,687 (100.0%) | Vocab: 37,000/37,000 | Done!
...

================================================================================
저장 중...
================================================================================

Saving tokenizer to artifacts/tokenizer_en.pkl...
✓ Tokenizer saved
Saving tokenizer to artifacts/tokenizer_de.pkl...
✓ Tokenizer saved
Saving vocabulary to artifacts/vocab.pkl...
✓ Vocabulary saved

✓ All artifacts saved to: artifacts
  - tokenizer_en.pkl
  - tokenizer_de.pkl
  - vocab.pkl
  - metadata.json
```

### 로드하여 실행

```
================================================================================
                    WMT14 Transformer Training
================================================================================

Configuration:
  Load directory: artifacts
  Save directory: artifacts
  Epochs: 100
  Max tokens per batch: 25,000
  Vocabulary size: 37,000
  Checkpoint directory: checkpoints

================================================================================
1. 데이터셋 로드
================================================================================
...

================================================================================
기존 결과물 로드 중...
================================================================================

Metadata:
  Created: 2026-01-30 14:23:45
  Vocab size: 37,000
  Shared vocab: True

Loading tokenizer from artifacts/tokenizer_en.pkl...
✓ Tokenizer loaded (vocab size: 37,000)
Loading tokenizer from artifacts/tokenizer_de.pkl...
✓ Tokenizer loaded (vocab size: 37,000)

Loading vocabulary from artifacts/vocab.pkl...
✓ Vocabulary loaded
  Source vocab size: 37,000
  Target vocab size: 37,000
  Shared: True

✓ All artifacts loaded from: artifacts

[토크나이저 훈련 완전히 스킵!]

================================================================================
어휘 정보
================================================================================
...
```

## 저장 파일 구조

### artifacts/ 디렉토리

```
artifacts/
├── tokenizer_en.pkl       # 21 MB - 영어 BPE 토크나이저
│   ├── vocab_size
│   ├── vocab (list)
│   ├── merges (dict)
│   └── base_tokenizer
│
├── tokenizer_de.pkl       # 22 MB - 독일어 BPE 토크나이저
│   └── (동일 구조)
│
├── vocab.pkl              # 15 MB - 어휘사전
│   ├── source_stoi (dict)
│   ├── source_itos (dict)
│   ├── target_stoi (dict)
│   ├── target_itos (dict)
│   └── shared (bool)
│
└── metadata.json          # 1 KB - 메타데이터
    ├── vocab_size
    ├── shared_vocab
    ├── src_pad_idx
    ├── trg_pad_idx
    ├── trg_sos_idx
    └── timestamp
```

**총 용량:** 약 60 MB

### checkpoints/ 디렉토리

```
checkpoints/
├── model_epoch_10.pt      # 250 MB - 10 에폭
├── model_epoch_20.pt      # 250 MB - 20 에폭
├── model_epoch_30.pt      # 250 MB - 30 에폭
├── ...
└── model_final.pt         # 250 MB - 최종 모델
```

각 체크포인트 포함 내용:
- model_state_dict
- optimizer_state_dict
- scheduler_state_dict
- epoch, train_loss, val_loss

## 고급 사용법

### 1. 다른 vocab 크기로 여러 버전 저장

```bash
# Vocab 10K
python demo_wmt14_saveable.py --vocab_size 10000 --save_dir artifacts_10k --epochs 1

# Vocab 20K
python demo_wmt14_saveable.py --vocab_size 20000 --save_dir artifacts_20k --epochs 1

# Vocab 37K (논문)
python demo_wmt14_saveable.py --vocab_size 37000 --save_dir artifacts_37k --epochs 1
```

나중에 원하는 버전 선택:
```bash
python demo_wmt14_saveable.py --load_dir artifacts_10k --epochs 100
```

### 2. 체크포인트에서 재개

```python
# 별도 스크립트 작성
checkpoint = torch.load('checkpoints/model_epoch_50.pt')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# 50 에폭부터 재개
for epoch in range(checkpoint['epoch'], 100):
    # ...
```

### 3. 토크나이저만 교체하여 실험

```python
# tokenizer_custom.py
from tokenizer_with_progress import BPETokenizer

# 커스텀 토크나이저로 훈련
custom_tokenizer = BPETokenizer(vocab_size=50000)
custom_tokenizer.train(corpus)

# 저장
import pickle
with open('artifacts/tokenizer_en.pkl', 'wb') as f:
    pickle.dump(custom_tokenizer, f)
```

## 성능 비교

### 토크나이저 학습 포함 (기존)

| 단계 | 소요 시간 |
|------|----------|
| 데이터 로드 | 2분 |
| **토크나이저 훈련 (EN)** | **4-6시간** |
| **토크나이저 훈련 (DE)** | **4-6시간** |
| 어휘 구축 | 5분 |
| Iterator 생성 | 30초 |
| **총 (학습 전)** | **8-12시간** |

### 저장된 결과물 로드 (개선)

| 단계 | 소요 시간 |
|------|----------|
| 데이터 로드 | 2분 |
| **토크나이저 로드** | **5초** ⚡ |
| Iterator 생성 | 30초 |
| **총 (학습 전)** | **3분** ⚡⚡⚡ |

**시간 절약: 약 99%** (8-12시간 → 3분)

## 문제 해결

### Q1: "FileNotFoundError: Missing required files"

**원인:** 지정된 디렉토리에 필수 파일이 없음

**해결:**
```bash
# 디렉토리 내용 확인
ls -lh artifacts/

# 필수 파일: tokenizer_en.pkl, tokenizer_de.pkl, vocab.pkl, metadata.json
# 없으면 --load_dir 없이 다시 실행
python demo_wmt14_saveable.py --save_dir artifacts
```

### Q2: "Vocab size가 예상과 다름"

**원인:** 다른 설정으로 저장된 결과물 사용

**확인:**
```bash
# metadata.json 확인
cat artifacts/metadata.json

{
  "vocab_size": 10000,  # ← 확인
  "shared_vocab": true,
  ...
}
```

**해결:** 올바른 디렉토리 사용 또는 재학습

### Q3: 토크나이저 로드 후 성능이 이상함

**원인:** 다른 데이터셋으로 학습된 토크나이저 사용

**해결:** 동일한 데이터셋으로 학습된 토크나이저 사용

### Q4: 저장 파일이 너무 큼

**정상:** 
- tokenizer_*.pkl: 약 20-25 MB 각
- vocab.pkl: 약 15 MB
- 총 약 60 MB

**줄이는 방법:**
```bash
# 더 작은 vocab 사용
python demo_wmt14_saveable.py --vocab_size 10000
```

## 베스트 프랙티스

### 1. 처음에는 작은 vocab으로 테스트

```bash
# Step 1: 빠른 테스트 (2-3시간)
python demo_wmt14_saveable.py \
    --vocab_size 10000 \
    --save_dir artifacts_10k_test \
    --epochs 5

# Step 2: 잘 작동하면 큰 vocab으로 (8-12시간)
python demo_wmt14_saveable.py \
    --vocab_size 37000 \
    --save_dir artifacts_37k_final \
    --epochs 1
```

### 2. 토크나이저만 먼저 학습

```bash
# 토크나이저만 학습 (Ctrl+C로 중단)
python demo_wmt14_saveable.py \
    --vocab_size 37000 \
    --save_dir artifacts_37k \
    --epochs 1

# Press Enter to start training 후 Ctrl+C
# 토크나이저와 어휘는 이미 저장됨!
```

### 3. 여러 실험을 위한 디렉토리 구조

```
experiments/
├── artifacts_37k/          # 논문 설정
├── artifacts_20k/          # 중간 크기
├── artifacts_10k/          # 빠른 테스트
│
├── exp1_baseline/          # 실험 1
│   └── checkpoints/
├── exp2_lr_tuning/         # 실험 2
│   └── checkpoints/
└── exp3_batch_size/        # 실험 3
    └── checkpoints/
```

```bash
# 모든 실험이 동일한 토크나이저 사용
python demo_wmt14_saveable.py --load_dir artifacts_37k --checkpoint_dir exp1_baseline
python demo_wmt14_saveable.py --load_dir artifacts_37k --checkpoint_dir exp2_lr_tuning
python demo_wmt14_saveable.py --load_dir artifacts_37k --checkpoint_dir exp3_batch_size
```

## 요약

### 주요 장점

1. **시간 절약:** 8-12시간 → 3분 (99% 절감)
2. **재현성:** 동일한 토크나이저로 여러 실험
3. **유연성:** 다양한 vocab 크기 사전 준비
4. **편의성:** 명령행 옵션으로 간편 제어

### 권장 사용법

1. **처음:** `--save_dir`로 저장
2. **이후:** `--load_dir`로 즉시 시작
3. **실험:** 다양한 옵션 조합

이제 토크나이저 학습을 한 번만 하고, 무한 반복 실험이 가능합니다! 🚀
