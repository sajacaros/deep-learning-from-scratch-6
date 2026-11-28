import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from codebot.model import GPT
from codebot.utils import get_device, get_vocab_size, codebot_paths, split_by_docs

# 설정
device = get_device()
vocab_size = get_vocab_size()          # 기본 2000 (책은 1000, VOCAB=1000으로 실행)
paths = codebot_paths(vocab_size)
data_path = paths['data']
tokenizer_path = paths['tokenizer']
model_save_path = paths['pretrain']

# 하이퍼파라미터
context_len = 256
batch_size = 32
learning_rate = 3e-4
max_iters = 20000
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = 4 * embed_dim
dropout_rate = 0.1
val_frac = 0.02        # 검증용으로 떼어 둘 문서 비율 (책 밖 보충)
eval_interval = 1000   # 검증 손실을 재는 간격
torch.manual_seed(0)

# 데이터셋 클래스
class TokenDataset(Dataset):
    def __init__(self, tokens, context_len):
        # 토큰열을 텐서로 변환하여 저장
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        # 꺼낼 수 있는 샘플 수 반환
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        # 입력: idx번째부터 context_len개의 토큰
        x = self.tokens[idx:idx+self.context_len]
        # 레이블: 한 칸 뒤에서 시작하는 같은 길이의 토큰열
        y = self.tokens[idx+1:idx+self.context_len+1]
        return x, y

# 데이터 준비
ids = np.fromfile(data_path, dtype=np.uint16).astype(np.int64)
train_ids, val_ids = split_by_docs(ids, end_token_id=vocab_size - 1, val_frac=val_frac)
print(f"어휘 {vocab_size}: 학습 토큰 {len(train_ids):,}, 검증 토큰 {len(val_ids):,}")
dataset = TokenDataset(train_ids, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델과 옵티마이저
model = GPT(
    vocab_size=vocab_size,
    max_context_len=context_len,
    embed_dim=embed_dim,
    n_head=n_head,
    n_layer=n_layer,
    ff_dim=ff_dim,
    dropout_rate=dropout_rate
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f"파라미터 수: {total_params:,} ({total_params/1e6:.1f}M)")

@torch.no_grad()
def evaluate(model, val_ids):
    """검증 토큰을 겹치지 않는 context_len 창으로 잘라 평균 손실(토큰당, nat)을 잰다."""
    model.eval()
    v = torch.tensor(val_ids, dtype=torch.long)
    n = (len(v) - 1) // context_len
    x = v[:n * context_len].view(n, context_len)
    y = v[1:n * context_len + 1].view(n, context_len)
    total, count = 0.0, 0
    for k in range(0, n, 64):
        xb, yb = x[k:k + 64].to(device), y[k:k + 64].to(device)
        logits = model(xb)
        total += F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='sum').item()
        count += yb.numel()
    model.train()
    return total / count

losses = []
val_losses = []   # (반복 횟수, 검증 손실)
data_iter = cycle(dataloader)  # 무한 반복
start = time.time()
pbar = tqdm(range(max_iters))

for i in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    if (i + 1) % eval_interval == 0 or i == 0:
        val_losses.append((i + 1, evaluate(model, val_ids)))
        tqdm.write(f"iter {i + 1}: train {loss.item():.4f}, val {val_losses[-1][1]:.4f}")

elapsed = time.time() - start

# 결과 저장
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(f"loss_pretrain{paths['suffix']}.png")

# 모델 저장
model.save(model_save_path)

# 비교용 기록 (ch03/10_vocab_compare.py가 읽는다)
with open(f"codebot/log_pretrain{paths['suffix']}.json", 'w') as f:
    json.dump({'vocab_size': vocab_size, 'params': total_params, 'train_tokens': len(train_ids),
               'val_tokens': len(val_ids), 'elapsed_sec': elapsed, 'losses': losses,
               'val_losses': val_losses}, f)