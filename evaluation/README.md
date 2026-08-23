# Evaluation pipeline

이 폴더는 세 종류의 근거를 분리한다.

1. **Pipeline health**: 출력 존재, 검색 후보 내 선택, self-match 방지, 쿼리/재시도 상한 등 코드 불변조건
2. **Semantic quality**: 무작위 위치의 블라인드 사람 평가와 선택적 LLM judge
3. **Legacy evidence**: 2025년 baseline-failure challenge set 80건 및 원응답 40문항 × 3명

`39/80`은 전체 정확도가 아니라 baseline 실패 집합에서의 correction rate다. `rating3`는 원응답이 아니므로 사용하지 않는다. 원응답인 `rating2`만 파생 데이터로 보존하며 관측 투표를 재배치하거나 보정하지 않는다.

전체 데이터와 사례집은 사용자 게시글/식별자를 포함하므로 로컬 전용이며 Git에서 제외한다. 원본 workbook을 별도 보관한 상태에서 `prepare`로 SHA-256 계보를 포함해 재생성한다. 공개 저장소에는 비식별화와 공개 허가를 마친 표본만 별도로 넣는다.

## Quick start

```powershell
python -m evaluation.cli prepare --source-xlsx <legacy-cases.xlsx> --ratings-xlsx <ratings.xlsx>
python -m evaluation.cli legacy-report
python -m evaluation.cli run --limit 3 --output evaluation/runs/pilot.jsonl
python -m evaluation.cli run-report --runs evaluation/runs/pilot.jsonl
python -m evaluation.cli export-blind --output evaluation/human/blind_form.csv --key-output evaluation/private/blind_key.jsonl
python -m evaluation.cli export-blind --runs evaluation/runs/pilot.jsonl --output evaluation/human/refactor_blind.csv --key-output evaluation/private/refactor_key.jsonl
```

실험 실행은 케이스마다 즉시 JSONL에 append되므로 중단 후 같은 명령으로 재개할 수 있다. 데이터·모델·프롬프트·코드·인덱스·의존성 fingerprint가 달라지면 기존 파일로의 resume을 거부한다. `judge`는 API 비용이 드는 선택 기능이며, 결측 쌍을 제외하고 baseline/refactor의 A 위치를 전체 사례에서 정확히 균형화한다. 모델·rubric·schema·구현·의존성 fingerprint와 위치 해독키도 저장한다.

기본값은 A/B 위치가 서로 완전히 뒤집힌 `F1`, `F2` 두 form이다. 평가자를 두 form에 균등 배정한다. 별도 경로로 생성한 key JSONL은 평가자에게 전달하지 않는다. 비교 양쪽 중 결측이 있는 사례는 위치가 노출되므로 설문에서 자동 제외된다.

## 해석 제한

- gold match/acceptable-set annotation이 아직 없으므로 Recall@k, MRR, nDCG는 계산하지 않는다.
- 자동 점검 통과율은 품질 정확도가 아니다.
- 새 모델 비교는 동일 케이스, 동일 seed, 고정 모델/프롬프트 버전으로 실행한다.
- 논문용 주 결과는 사람 평가를 우선하고 LLM judge는 보조 분석으로 둔다.
