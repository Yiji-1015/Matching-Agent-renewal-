# Matching Agent - LangGraph Refactor

석사학위논문의 Matching Agent를 현재 LangGraph 1.x 패턴에 맞춰 정리한 독립 실행 폴더입니다. 졸업 당시 노트북, 발표자료, 중간 실험 산출물은 포함하지 않고 리팩토링 코드와 실행에 필요한 파일만 모았습니다.

## 핵심 구조

```text
START
  -> Orchestrator
  -> optional Message Analyzer / Web Search
  -> Query Reformer
  -> Retrieve
  -> [TypeMatch | RoleMatch | PersonaMatch]  # parallel fan-out/fan-in
  -> Main Selector
  -> Evaluator
     -> success / retry limit: END
     -> fail: Query Reformer부터 제한된 재탐색
```

- 그래프가 필수 실행 순서를 보장합니다.
- 노드는 공유 상태를 직접 수정하지 않고 부분 업데이트만 반환합니다.
- TypeMatch, RoleMatch, PersonaMatch는 같은 LangGraph super-step에서 병렬 실행됩니다.
- Evaluator는 `Command(update=..., goto=...)`로 재시도와 종료를 제어합니다.
- Query Reformer는 Pydantic 스키마로 정확히 2개의 서로 다른 쿼리를 반환합니다.
- 검색 결과는 원문·재구성 쿼리별 rank와 FAISS score provenance를 보존합니다.

## 폴더 구조

```text
.
├── matching_agent/                      # LangGraph 구현
├── prompts/                             # 역할별 프롬프트
├── tests/test_workflow.py               # API 없는 회귀 테스트
├── Matching_Agent_LangGraph_Debug.ipynb # 노드별 응답 확인 노트북
├── run_demo.py                          # CLI 실행
├── requirements.txt
├── .env.example
├── 0522_data.xlsx                       # 연구 데이터
└── 0522_data_updated/                   # FAISS index.faiss / index.pkl
```

## 설치

Python 3.12 환경을 권장합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`.env.example`을 `.env`로 복사한 뒤 API 키를 입력합니다.

```dotenv
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

`OPENAI_API_KEY`는 필수입니다. `TAVILY_API_KEY`는 고유명사나 신조어를 해석하기 위해 Web Search 경로가 호출될 때 필요합니다.

## 실행

```bash
python run_demo.py --message "고등학생 밴드에서 기타리스트를 구합니다" --username user_a --show-trace
```

노드별 답변과 checkpoint를 자세히 보려면 `Matching_Agent_LangGraph_Debug.ipynb`를 실행합니다.

## 평가 파이프라인

`evaluation/`에는 평가 코드가 들어 있고, 현재 로컬에는 2025년 challenge set 80건·원 사람평가 120표·전체 사례집도 파생 생성되어 있습니다. 사용자 게시글과 식별자가 포함된 파생 데이터/보고서는 공개 검토 전까지 `.gitignore` 대상입니다.

```bash
python -m evaluation.cli legacy-report
python -m evaluation.cli run --limit 3 --output evaluation/runs/pilot.jsonl
python -m evaluation.cli run-report --runs evaluation/runs/pilot.jsonl
python -m evaluation.cli export-blind --output evaluation/human/blind_form.csv --key-output evaluation/private/blind_key.jsonl
```

연구 설계와 지표의 해석 제한은 `evaluation/README.md`에 정리했습니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

테스트는 외부 API를 호출하지 않고 다음을 확인합니다.

- 원문과 재구성 쿼리의 후보 통합 및 provenance 보존
- 재구성 쿼리 정확히 2개 제약
- 세 Selector의 병렬 결과 병합
- Evaluator 실패 후 제한된 재시도
- 결과가 없는 경우 retry limit 종료
- 실제 노드의 프롬프트 포맷과 전체 그래프 완료

## 데이터와 Git

이 폴더에는 로컬에서 바로 실행할 수 있도록 약 106MB의 `index.faiss`가 포함되어 있습니다. GitHub의 일반 파일 크기 제한을 넘을 수 있으므로 `.gitignore`에서 데이터 파일을 제외했습니다. 새 원격 저장소에 공개할 때는 다음 중 하나를 선택해야 합니다.

1. Git LFS로 FAISS 인덱스와 데이터셋 관리
2. 데이터 파일을 Release 또는 별도 저장소로 배포
3. 공개 가능한 원천 데이터와 인덱스 재생성 스크립트만 제공

과거 `.env`와 졸업 코드에 포함된 API 키는 이 폴더에 복사하지 않았습니다. 실제 키가 들어 있는 `.env`는 커밋하지 마세요.
