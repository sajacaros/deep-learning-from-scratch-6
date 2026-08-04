# ch09 사전학습 실행 개요

『밑바닥부터 시작하는 딥러닝 ❻』 9장 webbot 사전학습 실행 기록.

## 실행 개요

| 항목 | 값 |
|---|---|
| 대상 | `ch09/01_pretrain.py` (webbot 사전학습) |
| 실행 위치 | `gpu:/data1/sajacaros_work/dlfs6` |
| GPU | 4번 1장 (H100 80GB), `CUDA_VISIBLE_DEVICES=4` |
| 연산 정밀도 | bf16 autocast |
| 시작 | 2026-08-03 18:00:43 |
| 종료 | 2026-08-04 04:29:57 |
| 소요 시간 | 10시간 29분 09초 |
| 결과 | 100,000스텝 전량 완료, 에러·NaN 없음 |
| 로그 | `logs/ch09_pretrain.log` (26MB) |
| 실험 추적 | wandb offline — `wandb/offline-run-20260803_180043-wtc8vttk` |

### 실행 명령

```bash
CUDA_VISIBLE_DEVICES=4 \
HF_HOME=/data1/sajacaros_work/hf_cache \
TOKENIZERS_PARALLELISM=false \
WANDB_MODE=offline \
setsid nohup .venv/bin/python ch09/01_pretrain.py \
    < /dev/null > logs/ch09_pretrain.log 2>&1 &
```

재실행용 스크립트: `tools/launch_ch09.sh` (이미 실행 중이면 새로 띄우지 않음)

## 모델 구성

파라미터 수 **152,316,672개 (152.3M)**

| 항목 | 값 |
|---|---|
| 레이어 수 (`n_layer`) | 12 |
| 임베딩 차원 (`embed_dim`) | 768 |
| 어텐션 헤드 (`n_head`) | 12 |
| KV 헤드 (`n_kv_head`) | 4 (GQA) |
| FFN 차원 (`ff_dim`) | 2048 |
| 컨텍스트 길이 | 1024 |
| 어휘 크기 | 50,000 |
| RoPE theta | 10000 |

## 학습 설정

| 항목 | 값 |
|---|---|
| 총 스텝 (`max_iters`) | 100,000 |
| micro batch | 32 |
| gradient accumulation | 4 |
| 유효 배치 | 128 시퀀스 = 131,072 토큰/스텝 |
| 학습률 | 6e-4 (warmup 500스텝 후 코사인 감쇠) |
| gradient clipping | 1.0 |
| 평가 주기 | 1,000스텝 |
| 체크포인트 주기 | 10,000스텝 |

## 데이터

| 항목 | 값 |
|---|---|
| 학습 데이터 | `webbot/owt_train.bin` — 2,649,004,800 토큰 (5.3GB, uint16) |
| 검증 데이터 | `webbot/owt_valid.bin` — 64,496,066 토큰 (129MB) |
| 토크나이저 | `webbot/merge_rules.pkl` |
| 출처 | huggingface `koki0702/zero-llm-data` |
| 총 학습량 | 약 131억 토큰 (데이터셋 기준 약 4.9 에폭) |

## 결과

**검증 손실 10.9833 → 3.0318** (perplexity 약 20.7)

| 스텝 | 0 | 1,000 | 10,000 | 30,000 | 50,000 | 70,000 | 90,000 | 99,999 |
|---|---|---|---|---|---|---|---|---|
| val loss | 10.9833 | 4.2709 | 3.3951 | 3.1965 | 3.1245 | 3.0788 | 3.0432 | **3.0318** |

- 초반 1,000스텝에서 급락 후 완만하게 수렴
- 마지막 10,000스텝 개선폭 0.011 — 학습률이 6.03e-09까지 떨어져 사실상 수렴 완료
- 검증 손실이 끝까지 단조 감소 — 과적합 징후 없음
- 추가 개선은 스텝 증가보다 데이터 확대 또는 모델 규모 확대가 필요

## 산출물

| 파일 | 크기 | 비고 |
|---|---|---|
| `webbot/model_pretrain.pt` | 582MB | 최종 모델 |
| `webbot/model_pretrain_step{10000..90000}.pt` | 각 582MB × 9개 | 중간 체크포인트 |
| `webbot_val_loss.png` | 21KB | 학습 곡선 |

체크포인트 총 5.7GB. `/data1` 여유 9.1TB.

## 남은 작업

- [ ] wandb 동기화 — `wandb sync wandb/offline-run-20260803_180043-wtc8vttk`
- [ ] 미푸시 커밋 2개 — `main...origin/main [ahead 2]`
- [ ] untracked — `tools/launch_ch09.sh`, `wandb/`
- [ ] `ch09/02_generate.py`로 생성 품질 정성 확인

## GPU 반납

2026-08-04 04:29 학습 종료와 함께 GPU 4번 메모리 전량 반납 완료 (0 MiB / 0% 사용, 잔여 프로세스 없음).
