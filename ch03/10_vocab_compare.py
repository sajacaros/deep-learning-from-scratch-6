# 어휘 1,000 vs 2,000 CodeBot 비교 (책 밖 보충)
#
# 먼저 두 어휘로 사전 학습 → SFT → GRPO를 모두 돌려 둔다.
#   VOCAB=1000 python ch03/01_pretrain.py && VOCAB=1000 python ch03/03_sft.py && VOCAB=1000 python ch03/09_grpo.py
#   VOCAB=2000 python ch03/01_pretrain.py && ...  (2000도 같은 순서)
# 그다음 이 스크립트가 같은 검증 문서·같은 생성 설정으로 둘을 잰다.
#   python ch03/10_vocab_compare.py  → codebot/vocab_compare.json, codebot/vocab_compare_samples.txt
#
# 토큰당 손실은 어휘가 다르면 비교할 수 없다(2,000의 토큰 하나가 더 긴 글자 조각이다).
# 그래서 같은 검증 문서의 바이트 수로 나눈 "바이트당 비트(BPB)"를 함께 잰다.
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import ast
import json
import math
import re
import time
import numpy as np
import torch
import torch.nn.functional as F
from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device, codebot_paths, split_by_docs

device = get_device()
VOCABS = [int(v) for v in os.environ.get('COMPARE_VOCABS', '1000,2000').split(',')]
context_len = 256
val_frac = 0.02          # 01_pretrain.py와 같아야 같은 검증 문서가 된다
n_samples = 100          # "def"로 생성할 함수 수
max_new_tokens = 400     # 문서 평균은 약 110~120토큰이라 대부분 끝 토큰에서 멈춘다


@torch.no_grad()
def val_metrics(model, tok, val_ids):
    """검증 토큰을 겹치지 않는 256토큰 창으로 잘라 손실을 재고, 바이트당 비트로도 환산한다."""
    model.eval()
    v = torch.tensor(val_ids, dtype=torch.long)
    n = (len(v) - 1) // context_len
    x = v[:n * context_len].view(n, context_len)
    y = v[1:n * context_len + 1].view(n, context_len)
    total = 0.0
    for k in range(0, n, 64):
        logits = model(x[k:k + 64].to(device))
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), y[k:k + 64].to(device).view(-1),
                                 reduction='sum').item()
    targets = y.flatten().tolist()
    n_bytes = sum(len(tok.id_to_bytes[t]) for t in targets if t != tok.end_token_id)
    return {
        'loss_per_token': total / len(targets),            # nat / 토큰
        'bits_per_byte': total / math.log(2) / n_bytes,   # bit / 바이트 (종료 토큰 글자는 빼고 셈)
        'n_tokens': len(targets),
        'n_bytes': n_bytes,
        'bytes_per_token': n_bytes / len(targets),
    }


def generation_metrics(model, tok, seed=0):
    """'def'로 함수를 n_samples개 생성해 문법이 맞는 비율과 길이·속도를 잰다."""
    torch.manual_seed(seed)
    valid, chars, new_tokens, secs, samples = 0, [], [], 0.0, []
    for _ in range(n_samples):
        t0 = time.time()
        text = generate(model, tok, "def", max_new_tokens=max_new_tokens, temperature=1.0)
        secs += time.time() - t0
        try:
            ast.parse(text)
            ok = True
        except SyntaxError:
            ok = False
        valid += ok
        chars.append(len(text))
        new_tokens.append(len(tok.encode(text)) - len(tok.encode("def")))
        samples.append((ok, text))
    return {
        'syntax_valid_rate': valid / n_samples,
        'mean_chars': float(np.mean(chars)),
        'mean_new_tokens': float(np.mean(new_tokens)),
        'sec_per_sample': secs / n_samples,
        'chars_per_sec': sum(chars) / secs,
    }, samples


def calculate_reward(ground_truth, response):
    """09_grpo.py와 같은 판정: 마지막 정수가 정답이면 1."""
    matches = re.findall(r'(-?\d+)', response)
    return 1.0 if matches and int(matches[-1]) == ground_truth else 0.0


@torch.no_grad()
def arithmetic_accuracy(model, tok):
    """1+1 ~ 9+9의 81문제를 온도 0으로 풀린다 (09_grpo.py의 평가와 같다)."""
    correct, wrong = 0, []
    for i in range(1, 10):
        for j in range(1, 10):
            prompt = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
            response = generate(model, tok, prompt, temperature=0)
            if calculate_reward(i + j, response) > 0:
                correct += 1
            else:
                wrong.append(f"{i}+{j}")
    return correct / 81, wrong


results, sample_lines = {}, []
for vocab in VOCABS:
    paths = codebot_paths(vocab)
    tok = BPETokenizer.load_from(paths['tokenizer'])
    ids = np.fromfile(paths['data'], dtype=np.uint16).astype(np.int64)
    _, val_ids = split_by_docs(ids, end_token_id=tok.end_token_id, val_frac=val_frac)

    r = {'vocab_size': vocab}
    model = GPT.load_from(paths['pretrain'], device=device)
    r['params'] = sum(p.numel() for p in model.parameters())
    r['val'] = val_metrics(model, tok, val_ids)
    r['generation'], samples = generation_metrics(model, tok)
    print(vocab, r['val'], r['generation'])

    for stage in ['sft', 'grpo']:
        m = GPT.load_from(paths[stage], device=device)
        acc, wrong = arithmetic_accuracy(m, tok)
        r[f'{stage}_arith_acc'] = acc
        r[f'{stage}_arith_wrong'] = wrong
        print(vocab, stage, f"{acc:.3f}", wrong)

    with open(f"codebot/log_pretrain{paths['suffix']}.json") as f:
        r['pretrain_log'] = json.load(f)
    with open(f"codebot/log_grpo{paths['suffix']}.json") as f:
        r['grpo_log'] = json.load(f)
    results[vocab] = r

    sample_lines.append(f"===== vocab {vocab}: 'def'로 생성한 첫 5개 =====")
    for ok, text in samples[:5]:
        sample_lines.append(f"--- 문법 {'OK' if ok else 'X'} ---\n{text}")

# 어휘별로 따로 돌린 결과도 합칠 수 있게, 기존 파일이 있으면 덮지 않고 합친다
out = 'codebot/vocab_compare.json'
if os.path.exists(out):
    with open(out) as f:
        old = json.load(f)
    old.update({str(k): v for k, v in results.items()})
    results = old
with open(out, 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
with open(f"codebot/vocab_compare_samples_{'_'.join(map(str, VOCABS))}.txt", 'w') as f:
    f.write("\n".join(sample_lines) + "\n")
print("saved", out)
