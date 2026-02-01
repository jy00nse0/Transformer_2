this code is forked from https://github.com/hyunwoongko/transformer/tree/master and edited for self study

# WMT14 + DataLoader 실제 사용 예시

## 빠른 시작 (Quick Start)

### 1. 패키지 설치
```bash
pip install torch datasets transformers
```

### 2. 기본 사용 코드

```python
from tokenizer import BPETokenizer
from data_loader_improved import DataLoader

# ============================================================================
# STEP 1: DataLoader 초기화
# ============================================================================
loader = DataLoader(
    ext=('.en', '.de'),
    tokenize_en=None,  # 나중에 설정
    tokenize_de=None,
    init_token='<sos>',
    eos_token='<eos>'
)

# ============================================================================
# STEP 2: 데이터셋 로드
# ============================================================================

# Option A: Multi30k (작은 데이터셋, 테스트용)
train, valid, test = loader.make_dataset(dataset_name='bentrevett/multi30k')
# ~30,000 sentence pairs

# Option B: WMT14 (논문의 실제 데이터셋)
train, valid, test = loader.make_dataset(dataset_name='wmt14')
# ~4.5M sentence pairs

# ============================================================================
# STEP 3: BPE 토크나이저 훈련
# ============================================================================
print("Training BPE tokenizers...")

# 코퍼스 추출
en_corpus = [item['translation']['en'] for item in train]
de_corpus = [item['translation']['de'] for item in train]

# 토크나이저 훈련 (논문: vocab_size=37000)
bpe_en = BPETokenizer(vocab_size=37000)
bpe_de = BPETokenizer(vocab_size=37000)

bpe_en.train(en_corpus)
bpe_de.train(de_corpus)

# DataLoader에 설정
loader.tokenize_en = bpe_en.tokenize
loader.tokenize_de = bpe_de.tokenize

# ============================================================================
# STEP 4: 어휘 사전 구축 (논문 준수)
# ============================================================================
loader.build_vocab(
    train_data=train,
    min_freq=2,              # 최소 빈도 2 이상
    max_vocab_size=37000,    # 논문: EN-DE 37k tokens
    shared_vocab=True        # 논문: shared vocabulary
)

# 특수 토큰 인덱스 추출
src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)

print(f"Vocabulary size: {enc_voc_size}")

# ============================================================================
# STEP 5: 데이터 Iterator 생성 (논문 준수)
# ============================================================================
train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test,
    max_tokens=25000,  # 논문: 배치당 ~25,000 토큰
    device='cuda',     # GPU 사용
    num_workers=4      # 멀티프로세싱
)

print(f"Train batches: {len(train_iter)}")
print(f"Valid batches: {len(valid_iter)}")
print(f"Test batches: {len(test_iter)}")

# ============================================================================
# STEP 6: 배치 사용
# ============================================================================
for batch in train_iter:
    src = batch.src  # [variable_batch_size, src_len]
    trg = batch.trg  # [variable_batch_size, trg_len]
    
    # 모델에 입력
    # output = model(src, trg[:, :-1])
    # loss = criterion(output, trg[:, 1:])
    
    print(f"Batch shape: src={src.shape}, trg={trg.shape}")
    break
```

## 주요 특징

### 1. 토큰 기반 배치 생성
```python
max_tokens=25000  # 배치당 약 25,000 토큰

# 결과:
# - 짧은 문장 배치: 더 많은 문장 포함
# - 긴 문장 배치: 더 적은 문장 포함
# - 메모리 사용량 일정
```

### 2. 공유 어휘 (Shared Vocabulary)
```python
shared_vocab=True  # 논문의 EN-DE 방식

# 결과:
# - source와 target이 같은 어휘 공유
# - 메모리 효율적
# - 언어 간 토큰 일치
```

### 3. 어휘 크기 제한
```python
max_vocab_size=37000  # 논문 명세

# 결과:
# - 정확한 재현성
# - 모델 크기 제어
```

## 배치 동작 예시

### 토큰 수 기반 배치 생성

```
max_tokens = 25000

Batch 1 (짧은 문장들):
- 문장 수: 125개
- 평균 길이: 200 토큰
- 총 토큰: ~25,000 토큰

Batch 2 (긴 문장들):
- 문장 수: 50개
- 평균 길이: 500 토큰
- 총 토큰: ~25,000 토큰

→ 모든 배치가 비슷한 토큰 수 유지
```

### 길이 기반 정렬 (Bucketing)

```
정렬 전:
[길이10, 길이100, 길이20, 길이90] → 패딩 45%

정렬 후:
[길이10, 길이12, 길이15, 길이20] → 패딩 20%

→ 패딩 약 50% 감소
```

## 완전한 학습 코드

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformer import Transformer

# ============================================================================
# 모델 초기화
# ============================================================================
model = Transformer(
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    trg_sos_idx=trg_sos_idx,
    d_model=512,              # 논문: base model
    enc_voc_size=enc_voc_size,
    dec_voc_size=dec_voc_size,
    max_len=512,
    ffn_hidden=2048,          # 논문: d_ff
    n_head=8,                 # 논문: h=8
    n_layers=6,               # 논문: N=6
    drop_prob=0.1,            # 논문: P_drop=0.1
    device='cuda'
).to('cuda')

# ============================================================================
# Optimizer (논문 준수)
# ============================================================================
optimizer = Adam(
    model.parameters(),
    lr=1.0,              # learning rate schedule로 조정됨
    betas=(0.9, 0.98),   # 논문: β_1=0.9, β_2=0.98
    eps=1e-9             # 논문: ε=10^-9
)

# ============================================================================
# Learning Rate Scheduler (논문 준수)
# ============================================================================
def lr_schedule(step):
    # 논문: lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    d_model = 512
    warmup_steps = 4000
    return (d_model ** -0.5) * min(
        (step + 1) ** -0.5,
        (step + 1) * (warmup_steps ** -1.5)
    )

scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

# ============================================================================
# Loss Function (Label Smoothing)
# ============================================================================
criterion = nn.CrossEntropyLoss(
    ignore_index=trg_pad_idx,
    label_smoothing=0.1  # 논문: ε_ls=0.1
)

# ============================================================================
# 학습 루프
# ============================================================================
print("Starting training...")

for epoch in range(100):
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(train_iter):
        src = batch.src  # [variable_batch_size, src_len]
        trg = batch.trg  # [variable_batch_size, trg_len]
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, trg[:, :-1])  # Teacher forcing
        
        # Reshape for loss calculation
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg_reshape = trg[:, 1:].contiguous().view(-1)
        
        # Compute loss
        loss = criterion(output_reshape, trg_reshape)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}, Batch {batch_idx}, "
                  f"Loss: {loss.item():.4f}, LR: {lr:.6f}")
    
    # Validation
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in valid_iter:
            src = batch.src
            trg = batch.trg
            
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output_reshape, trg_reshape)
            val_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_iter)
    avg_val_loss = val_loss / len(valid_iter)
    
    print(f"\nEpoch {epoch}:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    
    # 모델 저장
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')
```

## 디버깅 및 검증

### 배치 정보 확인
```python
for i, batch in enumerate(train_iter):
    total_tokens = batch.src.shape[0] * batch.src.shape[1]
    pad_count = (batch.src == src_pad_idx).sum().item()
    padding_ratio = pad_count / total_tokens
    
    print(f"\nBatch {i+1}:")
    print(f"  Shape: {batch.src.shape}")
    print(f"  Sentences: {batch.src.shape[0]}")
    print(f"  Max length: {batch.src.shape[1]}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Padding ratio: {padding_ratio:.2%}")
    
    if i >= 4:
        break
```

### 어휘 사전 확인
```python
print(f"Source vocab size: {len(loader.source.vocab)}")
print(f"Target vocab size: {len(loader.target.vocab)}")
print(f"Shared vocab: {loader.source.vocab is loader.target.vocab}")

print("\nSpecial tokens:")
for token in ['<pad>', '<unk>', '<sos>', '<eos>']:
    idx = loader.source.vocab.stoi[token]
    print(f"  {token}: {idx}")

print("\nTop 10 tokens:")
for idx in range(4, 14):
    token = loader.source.vocab.itos[idx]
    print(f"  {idx}: {token}")
```

## 성능 최적화 팁

### 1. GPU 메모리에 맞게 조정
```python
# 16GB GPU
max_tokens = 25000

# 12GB GPU
max_tokens = 18000

# 8GB GPU
max_tokens = 12000
```

### 2. 멀티프로세싱
```python
train_iter, valid_iter, test_iter = loader.make_iter(
    train, valid, test,
    max_tokens=25000,
    device='cuda',
    num_workers=4  # CPU 코어 수에 맞게 조정
)
```

### 3. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_iter:
    src, trg = batch.src, batch.trg
    
    optimizer.zero_grad()
    
    with autocast():
        output = model(src, trg[:, :-1])
        loss = criterion(output.view(-1, output.shape[-1]), 
                        trg[:, 1:].view(-1))
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

## 문제 해결

### Q: OOM (Out of Memory) 에러
A: `max_tokens` 값을 줄이세요.
```python
max_tokens = 12000  # 25000에서 줄임
```

### Q: 학습이 너무 느림
A: `num_workers`를 늘리세요.
```python
num_workers = 8  # 4에서 늘림
```

### Q: 어휘 크기가 예상과 다름
A: `max_vocab_size`를 명시하세요.
```python
loader.build_vocab(
    train_data=train,
    min_freq=2,
    max_vocab_size=37000  # 명시적 제한
)
```

## 참고 자료

- 논문: "Attention Is All You Need" (Vaswani et al., 2017)
- 데이터셋: WMT14 English-German
- 어휘: BPE with 37,000 shared tokens
- 배치: ~25,000 tokens per batch
- 모델: Base (d_model=512, N=6, h=8)
