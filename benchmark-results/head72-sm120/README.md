# Head72 SM120 vision-attention experiment

See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for final six-pair results and limitations, and [reproduction/README.md](reproduction/README.md) for pinned inputs, server launch and replay commands. submission/ contains the qualified two-file patch and exact sources.

Recompute statistics:
```bash
python reproduction/analyze_pairs.py --input results/paired_metrics.json --output recomputed-statistics.json
```

The export includes twelve raw archive SHA256 values. Independent raw source/input/token/ownership audits were completed before export; full response archives remain retained separately. No weights, images, credentials or private server logs are redistributed. Portable clients were checked for input/source equivalence; measurements came from the frozen guarded runner.
