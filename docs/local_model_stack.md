# Local Offline Model Stack

This repository expects a first-pass offline stack under `models/` for local-only S1000D-RAG experiments:

| Role | Expected local artifact |
| --- | --- |
| Text LLM | `models/llm/qwen36-27b/Qwen3.6-27B-IQ4_NL.gguf` |
| Vision-language model | `models/vlm/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf` |
| VLM projector | `models/vlm/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf` |
| Embeddings | `models/embedding/bge-m3/` with `pytorch_model.bin`, tokenizer, and config files |
| Reranker | `models/reranker/bge-reranker-v2-m3/` with `model.safetensors`, tokenizer, and config files |

## Verification

Run the dependency-light verifier from the repo root:

```bash
python scripts/verify_local_models.py
```

Optional JSON manifest/report:

```bash
python scripts/verify_local_models.py --output /tmp/s1000d-local-model-manifest.json --format json
```

The verifier only inspects filesystem metadata. It does **not** import ML runtimes, load model weights, run inference, download files, or mutate Chroma indexes. It exits non-zero if required artifacts are missing; use `--no-fail` only for documentation or inventory contexts where an incomplete stack should not fail the command.

## Tracking policy

Model files are local-only operational artifacts and should remain untracked. The repository `.gitignore` ignores `models/` so downloaded GGUF, safetensors, PyTorch binaries, tokenizer files, and generated model caches are not committed.

## Benchmark status

No benchmark, latency, VRAM, accuracy, retrieval-quality, or runtime performance metrics are claimed yet. The manifest proves only that the expected local files/directories exist and records their sizes.

## Next gate

The next real runtime step is a separate smoke/eval gate that intentionally loads the selected models and exercises retrieval/generation with explicit measurement capture. That gate should be reviewed separately before running because it can consume GPU/CPU memory and may mutate runtime outputs if configured to do so.
