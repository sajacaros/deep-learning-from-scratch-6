# 8장. 모델 (도전편)

> **한 줄 요약** — WebBot은 StoryBot보다 훨씬 큰 모델이다. **스케일링 법칙·친칠라 법칙**으로 크기(약 150M)를 정하고, **Flash Attention**·**GQA**로 어텐션의 속도와 KV 메모리를 줄인다. 그 밖에 **MoE**, **torch.compile**, **컨텍스트 길이 확장(PI·YaRN)** 을 소개한다.

```mermaid
flowchart LR
    A["5장 개선된 GPT<br/>(StoryBot, 약 20M)"] --> B["8.1 모델 크기 선택<br/>친칠라: 토큰 ≈ 20 × 파라미터"]
    B --> C["8.2 Flash Attention<br/>SRAM 블록 처리"]
    C --> D["8.3 GQA<br/>쿼리 헤드가 KV 헤드 공유"]
    D --> E["webbot/model.py<br/>(약 150M)"]
    B -.-> F["8.4 MoE<br/>8.5 torch.compile<br/>8.6 PI · YaRN"]
    F -.->|"여유가 있으면 도전"| E
```

| 기법 | 해결하는 문제 | WebBot 코드에 반영? |
|---|---|---|
| 스케일링 법칙 / 친칠라 법칙 | 계산 예산을 모델 크기와 데이터양에 어떻게 나눌까 | ✅ 하이퍼파라미터 설정 |
| Flash Attention | 어텐션이 메모리 대역폭에 묶임(메모리 바운드) | ✅ `F.scaled_dot_product_attention` |
| GQA | KV 헤드 수만큼 늘어나는 K·V 메모리 | ✅ `n_kv_head=4`, `enable_gqa=True` |
| MoE | 계산량을 억제하면서 파라미터 수 늘리기 | ✗ 소개만 |
| torch.compile | 파이썬 인터프리터 오버헤드 | ✅ `ch09/01_pretrain.py`에서 사용 |
| PI / YaRN | 학습 때보다 긴 컨텍스트에서 성능 붕괴 | ✗ 소개만 |

---

## 8.1 모델 크기 선택

### 스케일링 법칙

LLM의 손실은 파라미터 수 $N$과 학습 토큰 수 $D$가 늘수록 **멱법칙(power law)** 을 따라 줄어든다. 줄어드는 정도까지 **예측할 수 있다**는 점이 핵심이다. 작은 모델로 실험한 결과로 큰 모델의 성능과 최적 설정을 미리 추정할 수 있다.

**학습 계산량** [Kaplan et al., 2020]

$$
C \approx 6ND
$$

- 순전파: 토큰 하나당 약 $2N$ FLOP (파라미터마다 곱셈 1번 + 덧셈 1번)
- 역전파: 그 두 배인 약 $4N$ FLOP

### 친칠라 법칙 — 예산 $C$를 $N$과 $D$에 어떻게 나눌까

$C$가 고정되면 $N$과 $D$는 서로 맞바꾸는 관계다.

| 연구 | 결론 |
|---|---|
| OpenAI [Kaplan et al., 2020] | 예산이 늘면 **주로 모델을 키워라** |
| DeepMind [Hoffmann et al., 2022] — **친칠라 법칙** | 모델과 데이터를 **함께** 키워라. 기준: **토큰 수 ≈ 파라미터 수 × 20** |

예: 7B 파라미터 → 140B 토큰, 70B 파라미터 → 1,400B 토큰.

### WebBot에 적용

7장에서 준비한 데이터 ≈ **2.6B 토큰** → 2.6B ÷ 20 ≈ **130M 파라미터**.

**표 8-1 모델 크기 설정**

| 항목 | StoryBot | WebBot |
|---|---|---|
| 어휘 크기 | 10,000 | 50,000 |
| 층 수 | 4 | 12 |
| 임베딩 차원 | 512 | 768 |
| 헤드 수 | 16 | 12 |
| **KV 헤드 수** | 16 | **4** (GQA) |
| FFN 차원 | 1,344 | 2,048 |
| 컨텍스트 길이 | 256 | 1,024 |
| 파라미터 수 | 약 20M | 약 150M |

- 150M은 계산값보다 조금 크지만 2.6B ÷ 150M ≈ **파라미터당 17토큰**이라 친칠라 기준에서 크게 벗어나지 않는다.
- FFN 차원 2,048 ≈ 768 × 8/3 (5.2절의 SwiGLU 규칙). StoryBot도 1,344 ≈ 512 × 8/3 (= 1,365)보다 조금 작은 값이다.

<details>
<summary>파라미터 수 직접 세어 보기 (WebBot, 책에 없는 보충 계산)</summary>

head_dim = 768 / 12 = 64, KV 차원 = 4 × 64 = 256

| 부분 | 계산 | 개수 |
|---|---|---|
| `embed` | 50,000 × 768 | 38.4M |
| `unembed` (공유 안 함) | 768 × 50,000 | 38.4M |
| 층당 어텐션 | W_q 768² + W_k 768·256 + W_v 768·256 + W_o 768² | ≈ 1.57M |
| 층당 SwiGLU | 3 × 768 × 2,048 | ≈ 4.72M |
| 12층 합계 | 12 × (1.57M + 4.72M) | ≈ 75.5M |
| **총합** (RMSNorm 무시) | | **≈ 152M** |

→ 절반 가까이가 임베딩·출력층이다. 어휘 50,000의 비용이 크다.
</details>

> 💡 **에포크와 반복** — 친칠라 법칙은 데이터를 한 번만 보는 상황을 전제한다. 같은 데이터를 **약 4에포크까지** 반복해도 성능 저하가 거의 없다고 알려져 있다 [Muennighoff et al., 2023]. 자원에 여유가 있으면 에포크 수나 모델 크기를 늘리고, 부족하면 줄이면 된다.

> 💡 **학습 비용 vs 추론 비용** — 친칠라 법칙은 *학습* 비용만 최적화한다. 실제 서비스에서는 추론이 계속 일어나고 그 비용은 모델 크기에 비례한다. 그래서 작은 모델을 기준보다 많은 데이터로 학습시키는 편이 장기적으로 경제적일 수 있다. Llama 3 70B는 **파라미터당 215토큰**(친칠라 기준의 약 10배)으로 학습됐다.

---

## 8.2 Flash Attention

### 문제: 어텐션은 메모리 바운드다

| GPU 메모리 | 용량 | 속도 |
|---|---|---|
| 디바이스 메모리 (DRAM, HBM) | 큼 | 느림 |
| 캐시 메모리 (SRAM) | 작음 | 빠름 |

어텐션은 연산량에 비해 메모리 접근이 잦아 **DRAM 전송 속도가 병목**이 된다(**메모리 바운드**). 일반 구현은 $C \times C$ 어텐션 맵 전체를 DRAM에 썼다가 다시 읽기 때문에 특히 불리하다.

### 해결: 블록 단위로 SRAM 안에서 끝낸다

**Flash Attention** [Dao et al., 2022]
- Q·K·V를 블록으로 잘라 각 블록을 **SRAM 안에서** 처리한다 → 어텐션 맵 전체를 DRAM에 쓰지 않는다.
- 소프트맥스는 **온라인 소프트맥스**로 계산해 블록 단위로 처리해도 **결과가 정확히 같다**(근사가 아니다).

> 💡 **온라인 소프트맥스** [Milakov & Gimelshein, 2018] — 일반 소프트맥스는 전체 원소의 최댓값과 합을 먼저 구해야 한다. 온라인 방식은 블록을 하나씩 보면서 최댓값 $m$과 합 $\ell$을 순차적으로 갱신하고, 최댓값이 바뀌면 이전 결과를 보정한다. 보정 식을 쓰면 다음과 같다(책에는 말로만 설명되어 있어 보충함).
>
> $$m_{\text{new}} = \max(m_{\text{old}},\ m_{\text{blk}}), \qquad \ell_{\text{new}} = \ell_{\text{old}}\, e^{m_{\text{old}} - m_{\text{new}}} + \sum_{j \in \text{blk}} e^{x_j - m_{\text{new}}}$$

### 파이토치에서 쓰기

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

- 하드웨어와 텐서 크기를 보고 Flash Attention 등 **적합한 구현을 자동 선택**한다.
- `is_causal=True` → 인과 마스크 자동 적용(2.6 어텐션 마스크). 5장의 `tril` + `masked_fill` + `softmax` + `matmul`을 한 줄로 대체한다.

---

## 8.3 Grouped-Query Attention (GQA)

멀티헤드 어텐션은 헤드마다 Q·K·V를 따로 가진다. 헤드를 늘리면 표현력은 오르지만 **K·V 메모리(=KV 캐시)도 비례해서 늘어난다**.

**GQA** [Ainslie et al., 2023]: 여러 쿼리 헤드가 **하나의 KV 헤드를 공유**한다.

<img src="images/fig08-01.png" alt="그림 8-1 멀티헤드 어텐션과 GQA 비교" width="600">

*그림 8-1 — 멀티헤드 어텐션은 쿼리 헤드와 KV 헤드가 1:1, GQA는 쿼리 헤드 2개가 KV 헤드 1개를 공유한다(각 사각형이 헤드 하나). K·V 메모리가 절반이 된다.*

| | 쿼리 헤드 | KV 헤드 | K·V 메모리 |
|---|---|---|---|
| 멀티헤드 어텐션 (MHA) | $H$ | $H$ | 1 (기준) |
| **GQA** | $H$ | $H_{kv}$ ($1 < H_{kv} < H$) | $H_{kv}/H$ |
| (참고) MQA | $H$ | 1 | $1/H$ |
| WebBot | 12 | 4 (쿼리 3개씩 한 그룹) | **1/3** |

(MQA는 책에 나오지 않지만 GQA의 극단적인 경우로 알아 두면 좋다.)

### 파이토치에서 쓰기

```python
hidden = F.scaled_dot_product_attention(
    q, k, v, is_causal=True, enable_gqa=True
)
# q: (B, n_head,    C, head_dim)
# k: (B, n_kv_head, C, head_dim)   ← 헤드 수가 더 적다 (n_kv_head < n_head)
# v: (B, n_kv_head, C, head_dim)
```

### `webbot/model.py`의 MultiHeadAttention

5장 KV 캐시 버전과 비교해 바뀐 부분만 추렸다.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, n_kv_head, head_dim, rope=None):
        ...
        self.W_q = nn.Linear(E, H * D, bias=False)
        self.W_k = nn.Linear(E, KV * D, bias=False)    # ⭐ KV 헤드 수만큼만
        self.W_v = nn.Linear(E, KV * D, bias=False)
        self.W_o = nn.Linear(H * D, E, bias=False)

    def forward(self, x, use_cache=False):
        Q = self.W_q(x).view(B, C, H, D).transpose(1, 2)    # (B, H,  C, D)
        K = self.W_k(x).view(B, C, KV, D).transpose(1, 2)   # (B, KV, C, D)
        V = self.W_v(x).view(B, C, KV, D).transpose(1, 2)   # (B, KV, C, D)
        ...  # RoPE(offset), KV 캐시 갱신은 5장과 동일

        if use_cache:
            # 캐시가 최대 길이를 넘으면 오래된 부분을 잘라냄
            max_cache_len = self.rope.cos_cache.size(0) if self.rope else 2048
            if self.k_cache.size(2) > max_cache_len:
                self.k_cache = self.k_cache[:, :, -max_cache_len:]
                self.v_cache = self.v_cache[:, :, -max_cache_len:]
                self.cache_offset = max_cache_len

        # Flash Attention + GQA
        is_causal = is_first_call if use_cache else True   # 디코드에서만 False
        hidden = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal,
                                                enable_gqa=True)
        ...
```

(실제 코드는 `use_cache` 분기 두 개로 `scaled_dot_product_attention`을 따로 호출한다. 동작은 같다.)

> ⚠️ **디코드에서 `is_causal=False`인 이유** — 디코드 단계의 Q는 길이 1, K는 길이 t다. 이때 `is_causal=True`를 주면 파이토치는 마스크를 **왼쪽 위 기준**으로 만들기 때문에 새 쿼리가 첫 번째 키만 보게 된다. 5장에서 디코드 단계에 마스크를 생략한 것과 같은 이유로 `False`를 준다. (파이토치 동작에 대한 보충 설명이다.)

> ⚠️ 캐시를 잘라낸 뒤에는 `cache_offset`이 `max_cache_len`에 고정되고, RoPE는 `offset + C > max_context_len`일 때 offset을 끝쪽으로 당긴다. 위치가 정확하지 않은 단순한 처리라서, 학습 길이를 넘는 생성은 품질이 떨어질 수 있다(→ 8.6).

### KV 캐시 메모리 비교 (보충 계산)

토큰 하나, 층 하나당 K·V 저장량 = $2 \times n_{kv} \times d_{head}$ 개의 값.

| WebBot (12층, head_dim 64, 1,024토큰, bf16 2바이트) | KV 헤드 | 토큰·층당 값 개수 | 전체 KV 캐시 (배치 1) |
|---|---|---|---|
| MHA였다면 | 12 | 2 × 12 × 64 = 1,536 | 약 37.7MB |
| **GQA (실제)** | 4 | 2 × 4 × 64 = 512 | **약 12.6MB** |

배치 크기와 컨텍스트 길이에 비례해 늘어나므로, 큰 모델·긴 문맥·많은 동시 사용자 환경에서 차이가 커진다.

---

## 8.4 Mixture of Experts (MoE)

일반 트랜스포머는 모든 토큰이 **같은 FFN 하나**를 지난다. **MoE** [Shazeer et al., 2017]는 FFN(=**전문가**, Expert)을 여러 개 두고, 토큰마다 **상위 k개(top-k)** 만 골라 계산한다.

<img src="images/fig08-02.png" alt="그림 8-2 MoE 구조" width="600">

*그림 8-2 — 전문가 4개 중 1개(top-1)를 쓰는 MoE. **라우터**(작은 신경망)가 확률을 내고, "We"는 FFN 2(p=0.65), "Like"는 FFN 1(p=0.8)로 보낸다. 출력에는 그 확률을 곱한다. (그림 출처: [Zhou et al., 2022])*

- **장점**: 파라미터 수(전문가 수)는 크게 늘리면서 토큰당 **계산량은 k개 전문가 분량**으로 억제한다.

**학습의 어려움**

| 문제 | 내용 | 대응 |
|---|---|---|
| 선택은 미분 불가 | top-k 선택은 O/X라 고르지 않은 전문가에 기울기가 흐르지 않는다. 강화 학습에서 보는 문제와 비슷하며 학습을 불안정하게 만든다 | (책에서는 문제 제기만) |
| 부하 쏠림 | 특정 전문가만 계속 선택 → 나머지는 학습이 안 됨 → 더 안 뽑힘 (악순환) | **보조 손실(auxiliary loss)**: 선택이 치우치면 페널티를 줘 고르게 쓰도록 유도 |

---

## 8.5 torch.compile

파이토치는 **동적 계산 그래프**라 유연하지만 연산마다 파이썬 인터프리터를 거치는 오버헤드가 있다. `torch.compile`(파이토치 2.0+)은 **JIT(Just-In-Time) 컴파일**로 이 오버헤드를 줄인다.

```python
model = GPT(...)
model = torch.compile(model)   # 감싸기만 하면 된다
```

- **첫 실행**에서 최적화(컴파일) → **두 번째부터** 빨라진다.
- 내부: 파이썬 바이트코드에서 계산 그래프 추출 → **연산 융합(operation fusion)**, 메모리 레이아웃 최적화 → 최적화된 코드로 변환.

> 💡 첫 스텝이 오래 걸리는 건 정상이다. 짧은 실험에서는 컴파일 시간이 오히려 손해일 수 있다.

---

## 8.6 컨텍스트 길이 확장

사전 학습은 고정된 컨텍스트 길이로 하지만, 추론에서는 대화가 길어지는 등 **더 긴 입력**이 들어올 수 있다. 학습 범위 밖의 위치가 들어오면 위치 인코딩이 처음 보는 값이 되어 **성능이 크게 떨어진다**. RoPE 기반 모델의 해결책 두 가지:

| 방법 | 아이디어 | 특징 |
|---|---|---|
| **PI** (Position Interpolation) [Chen et al., 2023] | 위치 인덱스를 학습 범위 안으로 **압축**. 4배로 늘리려면 위치를 1/4로: $m \to m/4$ | 적은 파인튜닝으로 길이를 크게 늘린다. 단, **모든 주파수**의 회전 속도가 똑같이 느려진다 |
| **YaRN** [Peng et al., 2023] | **고주파 성분은 거의 그대로**, **저주파 성분을 중심으로** 조정 | 가까운 토큰 사이의 세밀한 위치 구분을 지킨다 |

5.1절의 $\theta_k = \Theta^{-2k/d}$ 기준으로 보면:
- **고주파** = $k$가 작은 쌍 (빠르게 회전) → 가까운 토큰 사이의 미세한 위치 차이 담당
- **저주파** = $k$가 큰 쌍 (천천히 회전) → 먼 거리의 위치 관계 담당
- PI는 $m\theta_k \to (m/s)\theta_k$로 **모든 $k$를 똑같이** 늦춘다 → 고주파 정보까지 흐려진다. YaRN은 저주파 쪽만 주로 늘린다.

---

## 핵심 정리

- **스케일링 법칙**: 손실은 $N$, $D$에 대해 멱법칙으로 줄고 예측 가능하다. 학습 계산량은 $C \approx 6ND$.
- **친칠라 법칙**: 토큰 수 ≈ 파라미터 수 × 20. WebBot은 2.6B 토큰 → 약 150M 파라미터(파라미터당 17토큰). 추론 비용까지 생각하면 작은 모델을 더 오래 학습시키는 선택도 있다(Llama 3 70B: 215토큰/파라미터).
- **Flash Attention**: 블록 단위 SRAM 처리 + 온라인 소프트맥스로 DRAM 전송을 줄인다. 결과는 정확히 같다. 파이토치에선 `F.scaled_dot_product_attention`.
- **GQA**: 쿼리 헤드 여러 개가 KV 헤드 하나를 공유한다. WebBot은 12:4 → K·V 메모리 1/3. `enable_gqa=True`.
- **MoE**: 라우터가 토큰마다 top-k 전문가를 고른다. 파라미터는 늘리고 계산량은 억제한다. 기울기 단절과 부하 쏠림 → 보조 손실.
- **torch.compile**: JIT 컴파일로 인터프리터 오버헤드를 줄인다(연산 융합 등).
- **PI / YaRN**: RoPE 위치를 압축해 컨텍스트를 늘린다. YaRN은 고주파를 보존한다.
- WebBot은 5장 구현만으로도 동작하지만, 여유가 있으면 이 개선들에 직접 도전해 보자.

## 이전·이후 장과의 연결

| 이번 장의 내용 | 관련 장 |
|---|---|
| 기반 아키텍처(RoPE, SwiGLU, RMSNorm, KV 캐시) | 5장 모델 (응용편) |
| 인과 마스크 | 2.6 어텐션 마스크 |
| StoryBot 모델 설정 | 6장 |
| WebBot 학습 데이터 약 2.6B 토큰 | 7장 |
| 여기서 정한 `webbot/model.py`로 WebBot 학습 | 9장 |

## 복습 질문

1. $C \approx 6ND$에서 6은 어디서 나오는가?
2. 친칠라 법칙으로 2.6B 토큰에 맞는 모델 크기를 구해 보라. WebBot이 그보다 조금 큰 150M을 써도 괜찮은 이유는?
3. 서비스로 오래 쓸 모델이라면 친칠라 기준보다 작은 모델을 더 많은 데이터로 학습하는 편이 나을 수 있다. 왜 그런가?
4. 어텐션이 "메모리 바운드"라는 말의 뜻은? Flash Attention은 결과를 근사하는가?
5. WebBot(헤드 12, KV 헤드 4)에서 `W_k`의 출력 차원은? KV 캐시는 MHA 대비 몇 배인가?
6. KV 캐시의 디코드 단계에서 `scaled_dot_product_attention`에 `is_causal=False`를 넘기는 이유는?
7. MoE에서 보조 손실이 없으면 어떤 악순환이 생기는가?
8. PI가 고주파 성분까지 똑같이 늦추면 어떤 문제가 생기고, YaRN은 이를 어떻게 피하는가?
