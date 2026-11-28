# ch09 그림 목록

| 그림 | 제목 | 원본 (PDF 쪽) |
|---|---|---|
| [그림 9-1](fig09-01.png) | 데이터 병렬 개요(각 GPU에 같은 모델을 배치하고 전체 데이터를 나누어 처리) | 2, 3부.pdf p.142 |
| [그림 9-2](fig09-02.png) | ZeRO의 메모리 사용량(그림 출처: [Rajbhandari et al., 2019]) | 2, 3부.pdf p.144 |
| [그림 9-3](fig09-03.png) | Weights & Biases로 기록한 WebBot의 검증 손실 추이 | 2, 3부.pdf p.147 |
| [흐름도 (a)](fig09-flow-a.png) | 옵티마이저 스텝 하나: DDP(GPU 2장 예) + 기울기 누적 4번 + no_sync (노트용으로 새로 그림) | `ch09/flow_graph.py` |
| [흐름도 (b)](fig09-flow-b.png) | WebBot 파라미터 수로 계산한 GPU 한 장의 메모리: AdamW, Muon, ZeRO 단계별 (노트용으로 새로 그림) | `ch09/flow_graph.py` |
| [흐름도 (장 전체)](fig09-d01.png) | 9장 전체 흐름 (노트용으로 새로 그림, mermaid 대체) | `ch09/diagrams.py` |
