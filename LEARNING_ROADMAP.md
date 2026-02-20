# CUDA Learning Roadmap

A structured path from raw CUDA kernels to production inference libraries.

---

## Repo Structure

```
cuda-from-scratch/
├── README.md
├── CMakeLists.txt          # top-level build
├── docker/
│   └── Dockerfile          # CUDA dev environment
├── common/
│   ├── utils.cuh           # timing, error checking macros
│   ├── tensor.cuh          # simple tensor wrapper
│   └── benchmarks.cuh      # perf measurement helpers
├── 01_matmul/
├── 02_softmax/
├── 03_layernorm/
├── 04_attention/
├── 05_kernel_fusion/
├── 06_transformer/
├── 07_quantization/
├── 08_inference_engines/
└── notebooks/              # optional: Jupyter for profiling visualizations
```

---

## Phase 1 — CUDA Fundamentals + MatMul (`01_matmul`)

Start here to internalize the GPU memory hierarchy before anything else.

- `naive.cu` — baseline global memory matmul
- `shared_memory.cu` — tiled matmul with shared memory, understand bank conflicts
- `coalesced.cu` — memory coalescing, transpose tricks
- `cublas_baseline.cu` — benchmark against cuBLAS to know where you stand
- `wmma.cu` — Tensor Core matmul via WMMA API (stretch goal)

**Key concepts:** thread/block/grid hierarchy, warp divergence, occupancy, `__syncthreads()`, coalesced access patterns.

**Resources:** *Programming Massively Parallel Processors* (Kirk & Hwu), CUDA C++ Best Practices Guide.

---

## Phase 2 — Reduction Kernels (`02_softmax`, `03_layernorm`)

Reductions are the hardest primitive to get right and appear everywhere in transformers.

- `02_softmax/`: naive → warp-shuffle reductions → online softmax (numerically stable, single-pass)
- `03_layernorm/`: forward + backward, fused mean/variance computation

**Key concepts:** warp shuffle intrinsics (`__shfl_down_sync`), atomic ops, two-pass vs one-pass reductions, numerical stability.

**Resources:** warp primitives docs, "Optimizing Parallel Reduction" (Mark Harris).

---

## Phase 3 — Attention (`04_attention`)

- `naive_attention.cu` — unfused QKV matmuls + softmax
- `flash_attention_v1.cu` — implement the tiling algorithm from the paper (forward pass)
- Bonus: backward pass (very hard, very educational)

This phase synthesizes everything from Phases 1 and 2. Read the FlashAttention paper alongside your implementation.

**Resources:** FlashAttention paper (Dao et al.), Tri Dao's original FA1 code.

---

## Phase 4 — Kernel Fusion (`05_kernel_fusion`)

- `fused_bias_relu.cu` — simple warmup
- `fused_layernorm_linear.cu` — fuse layernorm + projection
- `fused_attention.cu` — pull your FlashAttention work here and clean it up
- Profiling with `ncu` (Nsight Compute): learn to read roofline analysis, identify memory vs compute bound kernels

**Key concepts:** fusion motivation (memory bandwidth savings), register pressure tradeoffs, `__launch_bounds__`.

**Resources:** Nsight Compute documentation, roofline model explainers.

---

## Phase 5 — Transformer from Scratch (`06_transformer`)

Assemble your kernels into a working GPT-2-small inference (forward pass first):

- Embedding lookup
- Multi-head attention (your fused kernel)
- FFN with fused bias + GELU
- LayerNorm
- Load HuggingFace GPT-2 weights and run inference to validate correctness

**Stretch goals:** KV cache, CUDA graphs for reducing launch overhead.

**Resources:** llm.c (Karpathy), *Making Deep Learning Go Brrrr* blog.

---

## Phase 6 — Quantization (`07_quantization`)

The natural bridge between your from-scratch transformer and production inference libraries. This is the piece your transformer won't have, and it's central to how llama.cpp achieves its performance.

- `int8_matmul.cu` — implement INT8 matmul with quantization/dequantization
- `int8_linear.cu` — quantized linear layer, compare accuracy vs FP32 baseline
- Study GGUF quantization formats (Q4_K_M, Q8_0 etc.) by reading llama.cpp's `ggml-cuda.cu`
- Benchmark memory bandwidth and throughput gains vs FP16

**Key concepts:** symmetric vs asymmetric quantization, per-tensor vs per-channel, absmax scaling, accuracy/performance tradeoffs.

**Resources:** "LLM.int8()" paper (Dettmers et al.), GGUF format specification, llama.cpp source.

---

## Phase 7 — Reading Production Inference Libraries (`08_inference_engines/`)

Use this directory for notes, annotated snippets, and experiments as you work through real codebases. By this point you'll have the intuition to read them rather than just use them.

### llama.cpp (start here)

Pure C/C++ with hand-written CUDA kernels, relatively self-contained, and very readable code. You'll recognize your own matmul and attention work directly.

- Start with `ggml-cuda.cu` — find your matmul and attention primitives in the wild
- Trace a single forward pass end-to-end through the codebase
- Study how quantized weights are unpacked and fed into kernels at runtime

**Goal:** understand how a lean, dependency-free inference engine actually works.

### vLLM (follow-up)

More ambitious. The key idea is **PagedAttention** — treating the KV cache like virtual memory pages to eliminate fragmentation and enable efficient continuous batching. Given your distributed systems background, the systems design here will likely be more interesting than the CUDA itself.

- Read the PagedAttention paper before touching the code
- Study `vllm/attention/` to see how paged KV cache is implemented
- Look at the continuous batching scheduler — this is the genuinely novel systems work

**Goal:** understand how a production-grade serving system handles memory management and batching at scale.

**Note:** the gap between Phase 5 and contributing meaningfully to vLLM is real. Treat llama.cpp as a learning step and vLLM as a destination you're building toward.

---

## Tooling

- **Build:** CMake with `enable_language(CUDA)`, one `CMakeLists.txt` per module
- **Profiling:** `ncu` for kernel-level, `nsys` for system-level — add profiling targets from day one
- **Correctness:** always have a CPU reference and assert against PyTorch with `torch.allclose()`
- **Docker:** pin your CUDA version (suggest 12.4) for reproducibility

---

## Summary Table

| Phase | Directory | Key Concept |
|---|---|---|
| 1 | `01_matmul` | Memory hierarchy, coalescing |
| 2 | `02_softmax`, `03_layernorm` | Warp reductions |
| 3 | `04_attention` | FlashAttention tiling |
| 4 | `05_kernel_fusion` | Bandwidth vs compute tradeoffs |
| 5 | `06_transformer` | End-to-end assembly |
| 6 | `07_quantization` | INT8, GGUF formats |
| 7 | `08_inference_engines` | llama.cpp → vLLM |
