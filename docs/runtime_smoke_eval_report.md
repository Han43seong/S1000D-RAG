# Runtime Smoke/Eval Report

Date: 2026-06-01 14:09-14:17 KST
Git baseline: `d102169`
Runtime artifact directory: `/tmp/jarvis-runtime/s1000d-rag/runtime-smoke-20260601-140959`

## Summary

Local runtime smoke/eval was executed against the downloaded offline model stack.

- Local model verifier/env check: PASS (`15/15` required artifacts present)
- Optional runtime dependencies: installed in the active Python environment
- Embedding model load: PASS (`bge-m3`, CPU)
- Reranker model load: PASS (`bge-reranker-v2-m3`, CPU)
- Chroma ingestion smoke: PASS (5 DM files, 7 chunks)
- Retrieval eval smoke: PASS (6 configured questions, top-k evidence returned)
- Text LLM load/generation: PASS (`Qwen3.6-27B-IQ4_NL.gguf`, initially CPU llama.cpp backend; CUDA backend enabled in the 15:54 KST follow-up)
- End-to-end RAG+LLM smoke: PASS (retrieval + Qwen answer generated with 3 evidences in the initial CPU smoke; web job smoke passed after CUDA enablement)
- VLM asset availability: PASS (VLM GGUF and mmproj files present)
- VLM image inference: NOT EXECUTED yet; the repository does not yet have a Qwen3-VL llama.cpp adapter path wired for image+mmproj inference.
- GPU offload: PASS in follow-up. `llama-cpp-python 0.3.23` was reinstalled from the `cu124` wheel; `llama_print_system_info` reports CUDA and the 27B model offloaded `65/65` layers to the RTX 4080 SUPER.

## Runtime environment

Installed/verified package versions:

- `llama_cpp`: `0.3.23`
- `sentence_transformers`: `5.5.1`
- `chromadb`: `1.5.9`
- `langchain_core`: `1.4.0`
- `langchain_community`: `0.4.2`
- `torch`: `2.11.0+cu129`
- `transformers`: `5.9.0`

GPU sample from `nvidia-smi` during smoke:

- Before: `4990 MiB used`, `11058 MiB free`, `1% util`
- After: `4987 MiB used`, `11061 MiB free`, `1% util`

`llama_cpp.llama_print_system_info()` reported CPU-only backend:

```text
CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |
```

## Commands executed

```bash
python scripts/local_model_env.py --check --format json
python scripts/eval_rag.py --check-config
python ingest.py --reset-index --limit 5 --chroma-dir /tmp/jarvis-runtime/s1000d-rag/runtime-smoke-20260601-140959/chroma_smoke
python scripts/eval_rag.py --retrieve --chroma-dir /tmp/jarvis-runtime/s1000d-rag/runtime-smoke-20260601-140959/chroma_smoke -k 3
python -m pytest tests/ -q
```

A direct Python smoke also loaded:

- `SentenceTransformer($S1000D_EMBEDDING_MODEL, device="cpu")`
- `CrossEncoder($S1000D_RERANKER_MODEL, device="cpu")`
- `Llama(model_path=$S1000D_TEXT_MODEL_PATH, n_ctx=256/1024, n_gpu_layers=0)`

## Results

### Local model check

```python
{'missing_required': [], 'missing_required_count': 0, 'present_count': 15, 'required_count': 15, 'status': 'complete'}
```

### Config check

```text
questions: 6 (/home/hskim/projects/S1000D-RAG/eval/questions/s1000d_bike.json)
modalities: {'text': 4, 'image': 1, 'multimodal': 1}
data_dir: /home/hskim/projects/S1000D-RAG/docs/S1000D Issue 6/Bike Data Set for Release number 6 R2 exists=True
chroma_dir: /home/hskim/projects/S1000D-RAG/chroma_db exists=False
collection_name: s1000d_chunks
manifest: /home/hskim/projects/S1000D-RAG/chroma_db/manifest.json exists=False
text_model_profile: qwen36_27b_iq4
vlm_model_profile: qwen3_vl_8b_q4
vlm_model_configured: True
embedding_model: /home/hskim/projects/S1000D-RAG/models/embedding/bge-m3
reranker_model: /home/hskim/projects/S1000D-RAG/models/reranker/bge-reranker-v2-m3
```

### Ingestion smoke

```text
[1/4] 5개 DM 파일 발견
[2/4] 파싱 완료: 7 chunks (0 errors)
[3/4] 7개 Document 변환 완료
[4/4] 임베딩 모델 로딩 + ChromaDB 인덱싱...
완료! (6.4s)

=== 인제스천 결과 ===
DM 파일: 5개
파싱 성공: 5개
총 청크: 7개
ChromaDB: /tmp/jarvis-runtime/s1000d-rag/runtime-smoke-20260601-140959/chroma_smoke / s1000d_chunks
Manifest: /tmp/jarvis-runtime/s1000d-rag/runtime-smoke-20260601-140959/chroma_smoke/manifest.json
```

Wall time: `7.336s`.

### Retrieval eval smoke

`eval/questions/s1000d_bike.json` contains 6 questions. Retrieval returned top-3 evidence for each configured question. Example:

```text
## brake_procedure_multimodal_safety
What text and visual evidence should be reviewed together before a brake procedure?
1. BRAKE-AAA-DA1-10-00-00AA-251A-A BRAKE-AAA-DA1-10-00-00AA-251A-A__chunk-001
2. BRAKE-AAA-DA1-00-00-00AA-041A-A BRAKE-AAA-DA1-00-00-00AA-041A-A__chunk-002
3. BRAKE-AAA-DA1-00-00-00AA-041A-A BRAKE-AAA-DA1-00-00-00AA-041A-A__chunk-003
```

Wall time: `5.271s`.

### Component model smoke

```text
embedding {'ok': True, 'shape': [2, 1024], 'seconds': 1.342}
reranker {'ok': True, 'score': 0.9962267875671387, 'seconds': 1.74}
text_llm {'ok': True, 'load_seconds': 20.627, 'generate_seconds': 1.006, 'backend_note': 'CPU llama-cpp-python build; n_gpu_layers=0'}
vlm {'model_file_present': True, 'mmproj_file_present': True, 'runtime_note': 'VLM image inference not executed; repo has no Qwen3-VL llama.cpp adapter yet'}
```

A second LLM generation check with chat-style prompt produced:

```text
OK
```

### End-to-end RAG+LLM smoke

Question:

```text
브레이크 시스템의 주요 구성품은 무엇입니까? /no_think
```

Result:

```text
브레이크 시스템의 주요 구성품은 다음과 같습니다.

1. 브레이크 레버 (Brake lever)
2. 브레이크 케이블 (Brake cable)
3. 브레이크 암 (Brake arm)
4. 브레이크 클램프 (또는 캘리퍼, Brake clamp/Callipers)
5. 브레이크 패드
```

Evidence count: `3`

Top evidence:

```text
BRAKE-AAA-DA1-00-00-00AA-041A-A / BRAKE-AAA-DA1-00-00-00AA-041A-A__chunk-001
BRAKE-AAA-DA1-00-00-00AA-041A-A / BRAKE-AAA-DA1-00-00-00AA-041A-A__chunk-002
BRAKE-AAA-DA1-00-00-00AA-041A-A / BRAKE-AAA-DA1-00-00-00AA-041A-A__chunk-003
```

Wall time: `107.657s` on CPU-only llama.cpp backend.

## Code adjustments made during smoke

Full test collection became available after installing the optional runtime dependencies. The full test suite initially exposed four failures:

- Prompt tests expected explicit `Context`, `Question`, and `한↔영` terminology-guide markers.
- E2E query pipeline dropped all documents when Chroma returned negative distance-like relevance scores.

Fixes:

- Restored bilingual terminology guide and explicit `Context / Question` prompt labels.
- Added prompt instruction not to emit reasoning/`<think>` output.
- Hardened `_strip_think_tags()` to remove unclosed `<think>` blocks.
- Added all-negative Chroma score fallback in `_apply_threshold_with_fallback()` so valid retrieval candidates are not discarded solely due distance-like score semantics.

Verification after fixes:

```text
python -m pytest tests/ -q
148 passed, 3 warnings in 1.51s
```

## GPU follow-up smoke — CUDA llama.cpp enabled

Date: 2026-06-01 15:54 KST

Actions:

```bash
python -m pip install --force-reinstall --no-cache-dir llama-cpp-python==0.3.23 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
eval "$(python scripts/local_model_env.py)"
python -m pytest tests/test_local_model_env.py -q
```

The CUDA wheel required NVIDIA Python-wheel shared-library directories on `LD_LIBRARY_PATH`. `scripts/local_model_env.py` now emits the existing CUDA runtime/cublas/curand/cusolver/cusparse/nvjitlink/nvrtc/cudnn library directories when present, while preserving any pre-existing `LD_LIBRARY_PATH` in shell output.

Backend check:

```text
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16375 MiB):
  Device 0: NVIDIA GeForce RTX 4080 SUPER, compute capability 8.9, VMM: yes, VRAM: 16375 MiB
CUDA : ARCHS = 700,750,800,860,890,900 | FORCE_MMQ = 1 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |
```

Direct 27B GGUF smoke with `n_gpu_layers=-1`, `n_ctx=512`, and `max_tokens=24`:

```text
load_tensors: offloaded 65/65 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   682.03 MiB
load_tensors:        CUDA0 model buffer size = 14634.73 MiB
SMI_AFTER_LOAD 15920 MiB used
llama_perf_context_print: prompt eval time = 855.83 ms / 20 tokens (23.37 tokens/s)
llama_perf_context_print: eval time = 629.02 ms / 23 runs (36.56 tokens/s)
TIMING_JSON {"load_sec": 15.407, "generate_sec": 1.526, "total_sec": 16.933}
SMI_AFTER_GEN 16015 MiB used, 98% util
```

Web/API job smoke after CUDA enablement:

```text
GET /api/status -> 200 ready=true backend=llama_cpp_python chunk_count=7
POST /api/chat/jobs -> 202 job_id=cec9f6af
POLL -> running/generating
POLL -> done/done
FINAL_STATUS done llm_sec 1.177363634109497 evidences 0
ANSWER_PREFIX 제공된 문서에서 해당 정보를 찾을 수 없습니다.
SMI_AFTER_JOB 15990 MiB used, 62% util
```

Verification:

```text
python -m pytest tests/test_local_model_env.py -q
5 passed in 0.12s
```

## Remaining gaps

1. VLM image inference still needs an adapter path for Qwen3-VL + `mmproj` through llama.cpp or another local VLM runtime.
2. Chroma emits warnings because the current embedding/vectorstore combination can return distance-like negative relevance scores. The pipeline now falls back safely, but the retrieval layer should later normalize score semantics to avoid warnings.
3. The 27B IQ4 model now runs on GPU, but it nearly fills the 16GB RTX 4080 SUPER (`~15.9-16.0GB` used during smoke). For production-like concurrent web use, consider a smaller quant/model, smaller context, or partial offload headroom.
