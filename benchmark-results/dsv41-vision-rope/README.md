# DeepSeek-V4.1 vision Q/K RoPE validation

These results compare upstream `5325cfabd757819e2e01bb7eb3b28c5b39b8cec9` with the three-file fused vision RoPE patch. Tested candidate tree: `46a97e33cf1080b7dc9dc8182a4c2a3d550363e0`. Original local commit: `ef3d0bfea1b8f2d7c18b3a4acc61a32cdff01237`. Publication may give the identical tree a different commit ID.

## Environment

- One node, eight SM120 GPUs with approximately 96 GiB each; driver reports NVIDIA RPBZZZ6 (owner identifies them as RTX PRO 6000).
- PyTorch 2.14.1+cu130, Triton 3.8.0, DeepGEMM 0.2.6, FlashInfer 0.7.2, Python 3.12. This was an existing local environment, not a pristine official container.
- DeepSeek-V4.1-Flash weights at revision `314113f8a7dadb388215396ba5fbee2c610a04d6`; TP8/EP8.
- Both arms used the flags below to avoid a metadata/dispatch startup inconsistency on the pinned base and SM120. That environment repair is excluded from the optimization claim. CUDA NVRTC libraries must also be discoverable in the local installation.

Launch each arm from its respective checkout with the same weights and dependencies:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SGLANG_VLM_CACHE_SIZE_MB=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 SGLANG_OPT_USE_TILELANG_INDEXER=0
export PYTHONPATH="$SGLANG_ROOT/python"
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name deepseek-v41-flash \
  --host 127.0.0.1 --port 18990 --tp-size 8 --ep-size 8 \
  --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 \
  --mem-fraction-static 0.72 --max-running-requests 16 --context-length 8192 \
  --trust-remote-code --disable-radix-cache --mm-preprocess-cache-size-mb 0
```

Set `SGLANG_ROOT` and `MODEL_PATH` to local paths. No idle GPU loads or diagnostic instrumentation were active during performance measurement.

## Sustained end-to-end performance

Use the 16 scikit-image sample files enumerated in `dsv41_sustained_probe.py` in an `IMAGE_DIR`. The script converts each to RGB, fits to 448×448 and encodes PNG. It cycles four images per request and sends the same prompt, with 64 output tokens. This is a synthetic request mix with caches disabled, not a diverse production workload.

```bash
python dsv41_sustained_probe.py --image-dir "$IMAGE_DIR" \
  --ports 18990 --workers-per-port 4 --warmup 30 --duration 180 \
  --size 448 --images-per-request 4 --max-tokens 64 > window.json
```

Run A-B-A-B with a fresh server for each arm; run the command twice per server with separate output files. Each variant therefore has four 180-second windows (720 seconds). Throughput counts successful completions within the measurement window; requests finishing during drain are retained separately. Each window has 30 seconds of warmup.

| Metric | Baseline | Fused |
|---|---:|---:|
| Successful measured requests | 1,636 | 1,688 |
| Pooled requests/s | 2.2722 | 2.3444 |
| Median of window p50, ms | 1,752.82 | 1,704.44 |
| Median of window p95, ms | 1,791.07 | 1,734.36 |
| Mean GPU utilization | 90.35% | 91.51% |

Observed throughput +3.18%, p50 −2.76%, p95 −3.17%. All 3,324 measured requests succeeded and produced 64 tokens; warmup/drain also had no request errors or empty outputs. `dsv41_current_long_summary.json` contains window summaries. Earlier 60-second measurements showed +2.94% throughput but +1.42% p95 latency; the longer measurements did not reproduce that tail-latency regression. One node and request mix do not establish a universal improvement. Final generated text was not bitwise stable in all comparisons; see accuracy checks below.

## Accuracy

```bash
python dsv41_prepare_mmmu.py --output-dir "$MMMU_DATA"
python dsv41_mmmu_eval.py --data-root "$MMMU_DATA" \
  --sglang-root "$BASELINE_ROOT" --arm baseline1
```

Dependencies include requests, Pillow, pyarrow and huggingface_hub. Always use the baseline checkout's `benchmark/mmmu/eval_utils.py` for scoring both implementations. Restart servers in A-B-A-B order and use unique arm names (`baseline1`, `candidate2`, `baseline3`, `candidate4`). The evaluator refuses to overwrite a run.

MMMU/MMMU validation revision `876ce5cb130f7f7e290ce4d9984357737d4db5cf`: 900 questions, 30 subjects, 43 multi-image questions, all images preserved. The adapted labeled-image API formatter is not an official leaderboard harness. Concurrency 4, temperature 0, seed 42, max_tokens 1024. Random parser guesses get no primary-score credit.

| Arm | Correct | Multi-image correct | HTTP errors | Truncated | Parser fallback |
|---|---:|---:|---:|---:|---:|
| A1 | 602/900 | 29/43 | 0 | 2 | 1 |
| B1 | 602/900 | 30/43 | 0 | 4 | 2 |
| A2 | 607/900 | 30/43 | 0 | 4 | 2 |
| B2 | 607/900 | 30/43 | 0 | 4 | 1 |

Both mean accuracies are 67.1667%. `audit.json` contains paired summaries; a 10,000-resample question-cluster bootstrap gives a 0.00 percentage-point difference with a [−0.50,+0.50] interval. Repeated runs do not create 3,600 independent questions. The 43-question multi-image subset is small.

Baseline repeats changed correctness on 17 questions; candidate repeats on 13. A question that was correct in both baseline runs and wrong in both candidate runs was investigated: serial reruns were correct 10/10 for each implementation; fixed-neighbor concurrency-4 reruns gave A five times and B five times for each implementation. Identical visual inputs produced identical ViT/aligner fingerprints on all eight replicas. Those instrumented diagnostics change scheduling and are not performance evidence or proof of universal equivalence.

The fixed union of seven truncated/parser-fallback cases was additionally evaluated with max_tokens 4096: baseline 0/7 with three still truncated; candidate 1/7 with none truncated; no HTTP errors. This supplement does not replace the primary scores or establish that all generation issues are solved.

## Kernel correctness and checks

```bash
PYTHONPATH="$SGLANG_ROOT/python" python "$SGLANG_ROOT/test/registered/kernels/ops/attention/test_dsv41_vision_rope.py"
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --padding 256 \
  --error-exitcode 86 python dsv41_kernel_memcheck.py --sglang-root "$SGLANG_ROOT"
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool initcheck --error-exitcode 86 \
  python dsv41_kernel_memcheck.py --sglang-root "$SGLANG_ROOT"
```

- Registered suite passed on GPU 0 and GPU 7: five tests / 20 subtests, including strided Q/K, head tails, 16-bit patterns with sampled angles, graph replay and fallback paths. HIP dispatch is mocked; no AMD execution claim.
- Production shadow validation: 26,880 same-input calls across eight replicas, bitwise-equal Q/K. Observed production shape: BF16 `[1521,16,64]`.
- Compute Sanitizer 2025.3 isolated module checks: 28 shape/dtype/layout cases, memcheck and initcheck each zero errors.
- Full framework-import memcheck emitted 43 `cuGetProcAddress_v2` invalid-value API diagnostics; reference-only import reproduced all 43 without invoking the candidate. Do not interpret the full-stack check as clean.
- Full `pre-commit run --all-files` passed; local self-review completed. Independent upstream review and CI are separate requirements.

## Related implementation comparison

Related upstream PR: https://github.com/sgl-project/sglang/pull/44451, head `1b002a1192130d38f36ae3475628df6263946378`. It uses JIT CUDA C++, separate Q and K launches, in-place writes and fast math. This candidate uses one Triton Q/K launch, separate outputs and disabled FP contraction. Both target the same operation.

To reproduce `compare_upstream_44451.py`, use a scratch checkout of the pinned baseline with the two kernel files from that PR head applied. Set `PYTHONPATH` to that scratch checkout's `python` directory; pass `--candidate-root` pointing to this candidate and `--output comparison.json`. This script imports the existing PR implementation from the scratch checkout; it does not include that author's source.

For contiguous `[1521,16,64]`, CUDA Graph, three measurements with randomized arm order and **identical two input resets included in every measured arm**:

| Dtype | Eager reference, µs | PR #44451, µs | This patch, µs |
|---|---:|---:|---:|
| BF16 | 55.425 | 11.705 | 7.911 |
| FP16 | 55.273 | 11.690 | 7.852 |

These are operator-plus-reset measurements, not isolated kernel latency and not an end-to-end comparison between the two PRs. Raw rows are in `upstream_44451_comparison.json`. The existing PR has small contiguous numerical differences from eager; its tolerances may intentionally allow them. Its strided wrapper also differs, but current production VisionAttention materializes contiguous Q/K before the custom rotary callback, so this does not demonstrate a current production failure.

## Provenance

Published scripts replace host-specific paths and model endpoint constants with CLI arguments; scoring and measurement logic are unchanged. These portable copies passed syntax checks but were not used for a new full-model rerun. `script_provenance.json` records original executed-script and published-script hashes. Raw local experiment outputs remain retained; these public artifacts contain summaries and reproduction scripts, not dataset images or private host paths.
