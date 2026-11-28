import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # 이 파일의 한 단계 위 디렉터리로 이동
sys.path.append('.')  # 현재 디렉터리를 파이썬 경로에 추가

import pickle
from codebot.tokenizer import train_bpe
from codebot.utils import get_vocab_size, codebot_paths

vocab_size = get_vocab_size()  # 어휘 크기: 기본 2000 (책은 1000, VOCAB=1000으로 실행)
text = open("codebot/tiny_codes.txt").read()
merge_rules = train_bpe(text, vocab_size)

# 학습된 병합 규칙을 파일에 저장
with open(codebot_paths(vocab_size)['tokenizer'], "wb") as f:   # merge_rules_2000.pkl
    pickle.dump(merge_rules, f)