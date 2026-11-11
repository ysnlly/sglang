# Qwen3.5-9B head72 vision attention on SM120

Six paired native serving comparisons completed on 2026-11-11. Primary throughput improved **18.9564%**, paired bootstrap 95% CI **[18.3677%, 19.2706%]**. All ten predeclared regression controls passed. This is the frozen eight-replica synthetic image workload; it is not a claim about every multimodal workload.

## Results

| Metric | Median paired change | 95% interval |
|---|---:|---:|
| large_multi_c4_rps | +18.9564% | [+18.3677%, +19.2706%] |
| small_single_c1_rps | +0.0744% | [-0.1965%, +0.3096%] |
| small_single_c4_rps | -0.0722% | [-0.3040%, +0.1111%] |
| small_multi_c1_rps | -0.0528% | [-0.2500%, +0.1447%] |
| small_multi_c4_rps | -0.1661% | [-0.5057%, +0.6385%] |
| large_single_c1_rps | +13.5372% | [+13.4176%, +13.6351%] |
| large_single_c4_rps | +19.1565% | [+19.0202%, +19.3033%] |
| large_multi_c1_rps | +16.0308% | [+15.9910%, +16.1120%] |
| mmmu64_rps | +3.0552% | [+2.1196%, +3.6975%] |
| large_multi_c4_p50_ms | -16.0134% | [-16.0828%, -15.9185%] |
| large_multi_c4_p95_ms | -15.9942% | [-16.0652%, -15.9782%] |

For throughput, positive means faster; for latency, negative means shorter. Confidence intervals are per metric, not simultaneous family-wide bounds. Sampling unit: pair. Six fixed orders AB/BA/BA/BA/AB/AB, 10,000 bootstrap draws, seed20261110 for primary (successive seeds for controls). Primary lower bound must exceed5%; every control's regression upper bound must be at most5%.

## Configuration and scope

Eight SM120 GPUs with approximately96GB VRAM each, eight TP1 replicas; PyTorch2.14.1+cu130, Triton3.8.0, driver580.159.04. Base14e395aa5e5eb00c6a3e3c755bf224221817dbe9. Each role starts fresh model processes; original and candidate use the same physical devices sequentially. One/eight224x224 images or one/six3840x2176 images, concurrency1/4 per replica;128 measured and32warmup requests per stratum. Separate900-question MMMU control fixes output64tokens; it is not the full-task accuracy evaluation. Total23,088 measured requests in12roles, zero contract violations.

Both arms: Triton vision/text/linear attention, same100MBVLMcache, radix/preprocessing caches disabled, eager vision, text decode CUDA Graph up to8, context65536, chunkedprefill8192, memfraction0.70, temperature0/seed42. Queue fill/drain and HTTP parsing are included in fleet throughput. Per-request latency excludes pending client queue time; it is not TTFT. Complete launch and replay commands are in reproduction/README.md. Input preparation is outside measured timing.

All12raw archives were independently checked for source/inputs/prompt-token identity, actual GPU ownership, fixed output token counts, cleanup, and metric recomputation. All11independent estimates/CIs match formal results exactly. The failed previous invocation is excluded. Its CPU preparation overlap is not part of these results. No weights/images are included in this package.

## Correctness and limitations

Independent FP64 operator tests passed on8GPUs. BF16 tolerances:atol.02+rtolBF16epsilon,relativeL2<=.01,andmax/L2errors<=1.05timesoriginal+1e-7. EightGPU production encoder bridge:216layer shadows/eightfull outputs matched original bitwise, without claiming universal bitwise equality. Full900-questionMMMU two-round crossover:1114/1800originalvs1121/1800candidate;delta+.3889pp,question-cluster95%CI[-.3333,+1.1111]pp,predeclared noninferioritymargin-.5pp. Truncation35vs36;errors0. No superiority claim.

Isolatedinitcheck passed on8GPUs. Memcheck retainedfiveexistingTritonloaderAPIprobeerrors/GPU,reproducedoriginal-only,withnootherreports;thisisnotunconditionallycleanmemcheck. NoA100/H100performanceclaim. Modelweightrevisionwasnotrecoverablefromdownloadmetadata;configurationandallfourweighthashesareprovidedintheinputprovenancerecord.
