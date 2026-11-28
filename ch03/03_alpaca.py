import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import json
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_vocab_size, codebot_paths

# 토크나이저 불러오기
tokenizer = BPETokenizer.load_from(codebot_paths(get_vocab_size())['tokenizer'])   # 기본 2000

# JSON 데이터 읽기
with open('codebot/tiny_codes_sft.json') as f:
    data = json.load(f)

# 첫 번째 샘플 꺼내기
item = data[0]
print(item)
# {'instruction': 'Hello', 'response': 'Hello. What can I help you with?'}

# 알파카 포맷으로 변환
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
print(text)
# ### Instruction:
# Hello
#
# ### Response:
# Hello. What can I help you with?<|endoftext|>

# 토큰화
ids = tokenizer.encode(text)
print(ids)
# [35, 35, 35, 962, 519, 117, 389, 58, 10, 846, 10, 10, 35, 35, 35, 752, 568, 58, 10, 846, 46, 840, 1674, 1053, 473, 708, 108, 112, 932, 657, 63, 1999]
# (어휘 1,000이면 34개: ' What'·' can'이 각각 두 토큰으로 갈라지고 끝 토큰은 999)