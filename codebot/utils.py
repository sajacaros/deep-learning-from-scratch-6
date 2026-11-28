import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=1000, temperature=1.0):
    model.eval()  # 평가 모드

    # 프롬프트 토큰화
    device = next(model.parameters()).device  # 파라미터가 있는 장치 확인
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    # 생성된 토큰을 저장할 변수
    generated_ids = ids.clone()

    # 토큰 생성 반복
    for _ in range(max_new_tokens):
        # 콘텍스트 길이를 초과하면 끝부분만 사용
        if ids.size(1) > model.max_context_len:
            ids = ids[:, -model.max_context_len:]

        # 다음 토큰 예측
        logits = model(ids)[:, -1, :]
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        # 종료 토큰이 생성되면 중단
        if next_id.item() == tokenizer.end_token_id:
            break

        # 생성한 토큰 추가
        ids = torch.cat((ids, next_id), dim=1)
        generated_ids = torch.cat((generated_ids, next_id), dim=1)

    # 디코딩하여 반환
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    return generated_text

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# ── 어휘 크기 실험 (책 밖 보충) ─────────────────────────────────
# 이 저장소는 어휘 2,000을 기본으로 쓴다(책은 1,000). 환경 변수 VOCAB으로 바꿀 수 있다.
#   python ch03/01_pretrain.py              → 어휘 2,000
#   VOCAB=1000 python ch03/01_pretrain.py   → 책과 같은 어휘 1,000
DEFAULT_VOCAB_SIZE = 2000


def get_vocab_size():
    import os
    return int(os.environ.get('VOCAB', DEFAULT_VOCAB_SIZE))


def codebot_paths(vocab_size):
    """어휘 크기별 파일 경로. 1000은 책·다운로드 파일과 같은 원래 이름, 그 밖에는 _2000처럼 접미사를 붙인다."""
    s = '' if vocab_size == 1000 else f'_{vocab_size}'
    return {
        'data': f'codebot/tiny_codes{s}.bin',
        'tokenizer': f'codebot/merge_rules{s}.pkl',
        'pretrain': f'codebot/model_pretrain{s}.pt',
        'sft': f'codebot/model_sft{s}.pt',
        'grpo': f'codebot/model_grpo{s}.pt',
        'suffix': s,
    }


def split_by_docs(ids, end_token_id, val_frac=0.02):
    """끝에서 val_frac만큼의 '문서'를 검증용으로 뗀다.

    토큰 수가 아니라 문서 수로 자르므로, 어휘 크기가 달라도 검증 문서가 똑같다.
    반환: (train_ids, val_ids)
    """
    import numpy as np
    ends = np.flatnonzero(ids == end_token_id)        # 각 문서 끝(종료 토큰) 위치
    n_val = max(1, int(len(ends) * val_frac))
    cut = ends[-n_val - 1] + 1                        # 검증 첫 문서의 시작
    return ids[:cut], ids[cut:]
