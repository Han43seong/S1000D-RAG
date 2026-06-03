# LangSmith RAG Smoke Runbook

Purpose: verify that the WinneAI/S1000D web RAG path is working end-to-end and that LangSmith receives traces for the run.

## Preconditions

- Local model artifacts are present: `python scripts/local_model_env.py --check --format json`
- `.env` contains LangSmith tracing settings:
  - `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY`
  - `LANGSMITH_TRACING=true` or `LANGCHAIN_TRACING_V2=true`
  - `LANGCHAIN_PROJECT=S1000D-RAG` or `LANGSMITH_PROJECT=S1000D-RAG`
- The default Chroma index exists at `chroma_db/` and has chunks.
- CUDA runtime env and demo LLM defaults are emitted by `scripts/local_model_env.py` when NVIDIA Python wheels are present.
  - `S1000D_LLM_N_CTX=2048`
  - `S1000D_LLM_MAX_TOKENS=256`

## Start the web server

```bash
cd /home/hskim/projects/S1000D-RAG
eval "$(python scripts/local_model_env.py)"
uvicorn app_web:app --host 127.0.0.1 --port 8000
```

Open the UI from Windows at:

```text
http://localhost:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/status
```

Expected:

- `ready=true`
- `backend=llama_cpp_python`
- `chunk_count > 0`
- model name like `Qwen3.6-27B-IQ4_NL`

## Run automated smoke

In a second terminal:

```bash
cd /home/hskim/projects/S1000D-RAG
eval "$(python scripts/local_model_env.py)"
python scripts/run_langsmith_smoke.py
```

By default this runs the first stable gate question. To try every configured smoke question:

```bash
python scripts/run_langsmith_smoke.py --limit 3
```

Fast one-question check:

```bash
python scripts/run_langsmith_smoke.py --limit 1
```

API-only check without LangSmith polling:

```bash
python scripts/run_langsmith_smoke.py --skip-langsmith
```

## What the script checks

For each question in `eval/questions/langsmith_smoke.json` it:

1. Confirms `/api/status` is ready.
2. Creates a web chat job with `POST /api/chat/jobs`.
3. Polls `GET /api/chat/jobs/{job_id}` until terminal state.
4. Fails if:
   - job status is not `done`
   - the job has an error
   - no evidence is returned
   - expected DMC substring is not present in returned evidence
5. Polls LangSmith project `S1000D-RAG` for recent runs and expects trace names such as:
   - `rag_pipeline`
   - `vector_search`
   - `enhance_query`
   - `build_context`

## Manual LangSmith review checklist

Open LangSmith project `S1000D-RAG` and inspect the latest trace.

Check:

- Root run is present for the smoke timestamp.
- Retriever step returns relevant DMC/chunk IDs.
- Rerank/filtering does not drop all valid candidates.
- Prompt context contains only retrieved S1000D evidence.
- Answer is grounded in the evidence.
- Latency is acceptable for demo use.
- Errors/warnings are not present in the trace.

## Current caveats

- The default automated smoke uses one stable gate question. Additional configured questions are useful for manual regression checks, but may expose prompt/retrieval issues that should be triaged separately.
- The 27B IQ4 text model nearly fills the 16GB RTX 4080 SUPER during GPU smoke. Keep one request at a time for demo runs.
- VLM image inference is not wired yet. Image/multimodal questions should remain separate from this text smoke until the VLM adapter path is implemented.
- Chroma relevance-score warnings may still appear in tests because some scores are distance-like rather than normalized 0-1 similarities.

## Recommended demo questions

```text
브레이크 시스템의 주요 구성품은 무엇입니까? /no_think
브레이크 패드 청소 절차를 순서대로 요약해줘. /no_think
브레이크 수동 테스트에서는 무엇을 확인해야 합니까? /no_think
```
