#!/usr/bin/env bash
# Reproduce every result from the prepared data (data/*.parquet). Each step resumes / skips finished outputs.
# Usage: bash run_all.sh                       (one GPU; about 12 hours on an RTX 5880 Ada and a 24-core CPU)
#        ARCHIVES=/path/to/downloads bash run_all.sh   also builds the full DoHBrw-2020 capture for the E4 supplement
set -euo pipefail
cd "$(dirname "$0")"
P=${PROCS:-3}
run() { echo "== $(date +%H:%M:%S) $*"; python "$@"; }
run reference/run_reference_verification.py --out results/reference_verification   # constructed-input reference verification (CPU, float64)
run scripts/run_main.py --until 4                         # E0 protocol: split, preprocessing, schema, reference detectors
run scripts/run_detector_search.py --stage search --procs 4   # E1 explained-detector configuration (validation only, then locked)
run scripts/run_detector_search.py --stage test --procs 4     # E1 locked configuration on test; checkpoints of the explained detector
run scripts/run_main.py                                   # E3: explainer tuning, registry, RQ1, A0/A/B/C/D
# FlowSCE refinement settings: results/protocol/plus_config.json, fixed on validation alerts (scripts/validation_compare.py)
run scripts/run_plus.py --stage test --procs "$P"         # E3: FlowSCE (restoration + repair + polishing) on the registered alerts
run scripts/run_graph_checks.py                           # E0 graph statistics, E2 naive self-loop update rule
run scripts/run_detectors.py --procs 4                    # E1 detection comparison (variants, E-GraphSAGE, CANDICE-detector)
run scripts/run_coupling.py --windows 600 60 10 1 --procs "$P"   # E4 time-window coupling sweep
if [ -n "${ARCHIVES:-}" ]; then
  run scripts/prepare_doh_full.py --archives "$ARCHIVES"  # E4 supplement: complete DoHBrw-2020 layer-2 capture
  run scripts/run_coupling.py --datasets doh_full --windows 60 10 1 --procs "$P" --tag coupling_fullcapture_doh
fi
run scripts/run_baselines.py --sections 1                 # CANDICE (tuned lambda): validation tuning only
run scripts/run_comparisons.py --stage tune --procs "$P"  # E5 CANDICE (paper configuration): learning rate on validation
run scripts/run_comparisons.py --stage gen --procs 11 --dice-cpu   # E5 generators (CANDICE, CANDICE tuned, DiCE via dice-ml on CPU)
run scripts/run_comparisons.py --stage min --procs "$P"   # E6 minimizers on the seeds of each generator (+ RHC)
run scripts/run_plus.py --stage seeds --procs "$P"        # E6 FlowSCE as a minimizer of the CANDICE and DiCE seeds
run scripts/run_comparisons.py --stage min-tuned --procs "$P"   # E6 minimizers on the CANDICE (tuned lambda) seeds
run scripts/run_plus.py --stage seeds-tuned --procs "$P"  # E6 FlowSCE on the CANDICE (tuned lambda) seeds
run scripts/run_comparisons.py --stage budget --procs "$P"  # E6 minimizer budget ablation
run scripts/run_plus_ablation.py --procs "$P"             # E6 FlowSCE refinement: component ablation and budget caps
run scripts/run_comparisons.py --stage summary            # E5/E6 tables and paired bootstrap
run scripts/run_candice_metrics.py --stage stability --procs 11 --cpu   # E7 IS@10 (five runs, CPU workers)
run scripts/run_candice_metrics.py --stage efficiency --procs 3  # E7 sequential time for 500 alerts
run scripts/run_candice_metrics.py --stage summary        # E7 ADA@k
run scripts/run_analysis.py --parts sensitivity cases    # E9 sensitivity, E11 case studies
run scripts/run_schema_ablation.py --seeds 11 23 37 53 71 # E10 projection ablation
run scripts/run_domain_checks.py                          # E10 byte-duration-rate relations
run scripts/run_plausibility.py                           # E10 percentiles of the changed values, kNN reference scale, case table
run scripts/run_constrained.py --procs "$P"               # E10 FlowSCE and LBS under a 1st-99th percentile domain
run scripts/run_alert_analysis.py                         # per-alert coupling, stale search with a final check, false vs true alerts
run scripts/run_analysis.py --parts runtime              # E8 runtime scaling (run on an otherwise idle GPU)
run scripts/profile_oracle.py                             # sequential vs batched oracle throughput
run scripts/make_paper_tables.py                          # T1-T8, A1-A6, F1, results/paper/numbers.json
run scripts/summarize_results.py                          # results/SUMMARY.md
run scripts/audit.py                                      # exits 1 on any FAIL
