# Native eight-replica reproduction client

This folder reproduces the eight native image stress strata of the completed comparison. It does not replace the complete MMMU task qualification, the 900-request MMMU fixed-token control, the owned-process/resource audit or the six independent paired measurements. Formal results come from the frozen driver, not this new command-line wrapper. No performance result from this wrapper is claimed.

`native_stress_trace.py`, `mmmu_fixed_replay.py` and `frozen_fixed_trace.py` are byte-for-byte copies of the sources embedded in the frozen driver. See `source_provenance.json`. `replay_native.py` exposes input and endpoint arguments and preserves the original warmup prefix, 128 measured requests per stratum and fixed 64-token policy. Preparation-only mode sends no requests and imports no GPU framework. It verifies exact trace hashes, including image bytes, prompts, request order and payload hashes. A mismatch fails rather than silently substituting another workload.

## Environment and inputs

Measured base: `14e395aa5e5eb00c6a3e3c755bf224221817dbe9`; candidate kernel SHA256 `21e884372d5e9ae80a44873d0c8cce5b876ab925cb53213f615bef9cd0bcff5e`. Eight SM120 GPUs, eight separate TP1 replicas, PyTorch 2.14.1+cu130, Triton 3.8.0, CUDA driver 580.159.04. Client used Pillow 13.0.0 and requests. Use a compatible SGLang environment for this exact base and keep both arms identical apart from the reviewed kernel change.

Set `SOURCE` to the desired source checkout and `MODEL` to the Qwen3.5-9B weight directory. Both implementations must serve as `qwen35-9b`; this name is part of the frozen request. `IMAGES` must contain the sixteen named source files listed in `input_images.json`, originally from the existing local scikit-image data directory. Image data is not redistributed by this package. The file hashes are authoritative; different data or PNG encoders may produce a trace mismatch.

Obtain those exact public source files with `python download_native_images.py --output ./native-images`, then set `IMAGES=./native-images`. The downloader pins scikit-image commit `e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae` (v0.25.2), checks every measured source hash and image dimension, and writes a download receipt. All sixteen hashes match the [upstream registry](https://github.com/scikit-image/scikit-image/blob/e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae/skimage/data/_registry.py). Image origins and attribution are documented in the [upstream data loader](https://github.com/scikit-image/scikit-image/blob/e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae/skimage/data/_fetchers.py). This establishes an accessible byte-identical source; it does not establish which scikit-image release originally populated the measured host's cache.

The stress images are fitted to 224×224 or 3840×2176. This synthetic image workload measures execution cost, not image quality or a distribution of customer traffic. The 100 MB VLM cache is deliberately retained in both arms, so the request image count is not necessarily the physical encoder batch size. Actual source profiling established the attention calls in this setting.

`model_input_provenance.json` records configuration hashes, file sizes and full SHA256 hashes of all four weight shards from the completed quality input audit. That audit's hash is bound by the frozen performance contract and checked before each arm, together with current configuration hashes and weight file sizes/modification times. Weight bytes are not rehashed for every performance arm. The preexisting model directory has no download metadata identifying an upstream revision; no such revision is inferred from the hashes.

## Launch each replica

On the measured 180-logical-CPU host, replicas 0–3 use CPU ranges 0–19, 20–39, 40–59 and 60–79; replicas 4–7 use 90–109, 110–129, 130–149 and 150–169. Adjusting topology on another host changes the environment and must be disclosed. Run the following once per GPU with `GPU=0` through `GPU=7`, in separate terminals or under a process supervisor. Preserve the exact source selected for each arm. The command intentionally remains in the foreground so the owner can stop that replica without a broad process kill.

```bash
GPU=0
PORT=$((19400 + GPU))
if [ "$GPU" -lt 4 ]; then
  FIRST=$((GPU * 20))
else
  FIRST=$((90 + (GPU - 4) * 20))
fi
LAST=$((FIRST + 19))
env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$SOURCE/python" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  SGLANG_VIT_ENABLE_CUDA_GRAPH=0 SGLANG_VLM_CACHE_SIZE_MB=100 \
  SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python \
  CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13 FLASHINFER_DISABLE_VERSION_CHECK= \
  TRITON_CACHE_DIR="/dev/shm/VinfraT-qwen35-e2e-triton-$GPU" \
  TORCHINDUCTOR_CACHE_DIR="/dev/shm/VinfraT-qwen35-e2e-inductor-$GPU" \
  taskset -c "$FIRST-$LAST" python -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name qwen35-9b \
  --host 127.0.0.1 --port "$PORT" --tp 1 --dtype bfloat16 \
  --context-length 65536 --mem-fraction-static 0.70 \
  --attention-backend triton --linear-attn-backend triton \
  --mm-attention-backend triton_attn --cuda-graph-max-bs-decode 8 \
  --cuda-graph-backend-prefill disabled --max-running-requests 4 \
  --chunked-prefill-size 8192 --random-seed 42 --disable-radix-cache \
  --mm-preprocess-cache-size-mb 0 --mm-feature-transport cpu
```

Wait for all eight services to finish startup and verify `/health` and `/server_info`. The wrapper does not establish source identity or enforce exclusive ownership of the GPUs; preserve those records separately as the formal driver does. Do not run a second benchmark or profiler concurrently.

## Prepare or replay

From this folder, an input-only check is:

```bash
python replay_native.py --images "$IMAGES" --prepare-only --output ./prepared-inputs
```

The inference command sends 32 warmup requests followed by 128 measured requests for each of eight strata:

```bash
python replay_native.py --images "$IMAGES" --output ./native-reference \
  --urls http://127.0.0.1:19400/v1/chat/completions \
         http://127.0.0.1:19401/v1/chat/completions \
         http://127.0.0.1:19402/v1/chat/completions \
         http://127.0.0.1:19403/v1/chat/completions \
         http://127.0.0.1:19404/v1/chat/completions \
         http://127.0.0.1:19405/v1/chat/completions \
         http://127.0.0.1:19406/v1/chat/completions \
         http://127.0.0.1:19407/v1/chat/completions
```

Output directories must not exist. Raw responses retain native prompt/output token IDs and token usage. Fleet throughput is successful requests divided by one common fleet wall timer, including queue fill/drain, HTTP transfer and response validation. Preparation and evidence writes are outside that timer. Per-request latency starts when a client worker begins its request; it excludes waiting for that worker in the submitted client queue. Do not call it TTFT or average eight individual replica throughputs.

For a comparison, start a fresh fleet for each arm, use the same GPUs and inputs, retain raw records and compare prompt hashes. The formal experiment freezes six pairs in AB/BA/BA/BA/AB/AB order. A single run of this client does not establish repeatability, statistical acceptance or task accuracy.

## Completed evidence

See ../EXPERIMENT_REPORT.md and ../results/ for the six-pair results. No separate GPU run of the portable wrappers is claimed.

## Complete MMMU inputs and fixed-token control

`prepare_mmmu.py` uses the measured dataset export's RGB/PNG conversion and sorts the same 900 questions from 30 pinned parquet files. `mmmu_contract.json` binds their upstream revision and file hashes, plus a semantic hash covering all questions, answers, options and 982 image hashes. Only filesystem paths change to relative paths. The 43 multi-image questions retain every original image.

The export requires requests2.34.2, Pillow13.0.0, pyarrow26.0.1 and huggingface-hub1.33.0. Exact PNG byte reproduction additionally requires the measured Pillow zlib1.2.11 encoder. A stock macOS Pillow13 wheel used zlib-ng and produced different PNG hashes; the strict check rejected that export. Rebuilding Pillow against a private zlib1.2.11 restored all 982 hashes. This local reproduction setup does not modify the GPU server or contribute to its speedup. For nonstandard library locations, follow the [official Pillow source-build instructions](https://pillow.readthedocs.io/en/stable/installation/building-from-source.html). Do not relax the input hashes to accept different encoded payloads.

```bash
python prepare_mmmu.py --output ./mmmu-data
python replay_mmmu.py --data ./mmmu-data --prepare-only --output ./mmmu-input-check
```

The download is pinned to revision876ce5cb130f7f7e290ce4d9984357737d4db5cf. `--cache-dir` reuses a previously downloaded cache. No model requests are sent by those two commands. The fixed-token control can then be replayed against the same eight endpoints:

```bash
python replay_mmmu.py --data ./mmmu-data --output ./mmmu64-reference \
  --urls http://127.0.0.1:19400/v1/chat/completions \
         http://127.0.0.1:19401/v1/chat/completions \
         http://127.0.0.1:19402/v1/chat/completions \
         http://127.0.0.1:19403/v1/chat/completions \
         http://127.0.0.1:19404/v1/chat/completions \
         http://127.0.0.1:19405/v1/chat/completions \
         http://127.0.0.1:19406/v1/chat/completions \
         http://127.0.0.1:19407/v1/chat/completions
```

This sends all900 requests, fixed64 output tokens, seed20261110 for the frozen permutation and concurrency4 per replica. Its complete trace hash63dd7346e4fe92779e0413bc558c0e3595b8e2cd509d7c4af06657f08800a1c6 was reproduced locally exactly. In the formal driver this control follows the eight native image strata, with the same fresh fleet. It is a performance control, not an accuracy test.

## Complete-task accuracy client

`quality_client.py` preserves the measured evaluator's request bodies, grader, warmup and crossover scheduler through its byte-identical source file. `portable_mmmu.py` changes path resolution and endpoint selection, defers scorer imports for input-only checks, and pins both scorer files by SHA256. All900 max16384-token request hashes matched the saved formal quality run. A normalized AST comparison confirms that the request method differs only in the endpoint expression. These are input/code equivalence checks, not another model-quality run.

Use the complete SGLang evaluation environment, including its scorer dependencies, and the pinned source checkout. Start eight replicas with the server settings above. In round0, original runs on GPUs0,2,4,6 and candidate on1,3,5,7. In round1, start fresh services and reverse those roles. Both use `triton_attn`. Each question retains its physical GPU pair. The output labels are `default` for original and `split` for candidate.

Run `quality_client.py --data ./mmmu-data --source "$SOURCE" --round 0 --output ./quality/round0 --urls <the same eight endpoints>`, then repeat with `--round 1 --output ./quality/round1` after replacing the services with the reversed assignment. The token cap is fixed at16384, temperature0, seed42, thinking disabled and concurrency4 per replica. The warmup rejects request errors or truncated answers.

The client saves all raw answers and preserves the pinned upstream parser. Random-fallback guesses receive no primary-score credit. `completed_summary.json` additionally excludes truncated answers; `summary.json` retains the evaluator's uncapped-completion correctness count for inspection. Do not interchange those metrics. The wrapper does not establish service ownership, cleanup, source identity, or a complete qualification verdict. Those records and the frozen question-cluster statistical analysis accompany the actual reported experiment separately.

CPU validation also rejected modified image bytes, a changed question with its file hash recomputed, and duplicate endpoints. See `quality_input_equivalence.json`, `mmmu_negative_controls.json` and `mmmu-preparation-check/trace_manifest.json`. The initial failed macOS export and build logs remain in the local research workspace; no measured input, threshold or GPU code was changed to obtain these checks.
