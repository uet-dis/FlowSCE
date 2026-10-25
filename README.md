# FlowSCE: Policy-Consistent Counterfactual Reasoning for Graph-Based Network Intrusion Detection

Counterfactual explanations for graph-based intrusion detectors through exact flow-to-host policy lifting, total-gradient
search and validity-preserving refinement on a shared reconstructed oracle.

## Abstract

Graph-based intrusion detectors derive host representations from network flows, so modifying a queried flow may also
change the contextual inputs used by the detector. Counterfactual explanations that freeze these dependencies can therefore
describe an inconsistent model state. We introduce FlowSCE, a policy-consistent counterfactual reasoning framework that
treats graph construction as part of the frozen prediction pipeline. For affine flow-to-host aggregation, FlowSCE provides
exact intervention lifting, total differentiation, and a score-discrepancy analysis, while exposing a common reconstructed
oracle to compatible search and refinement procedures. Experiments on CIC-IDS2018, DoHBrw-2020, and UNSW-NB15 show that
stale context can produce nominal label flips that fail reconstructed validation, particularly under strongly coupled
short-window policies, and that omitting dependency gradients can reduce counterfactual success. Across 1,500 primary alert
evaluations, FlowSCE returns 1,499 policy-consistent counterfactuals and substantially reduces their support through
validity-preserving refinement. The results also show that a lower-cost adapted minimizer can achieve comparable sparsity
and that some sparse contrasts remain distributionally extreme or violate unenforced traffic relations. FlowSCE therefore
improves the computational validity of counterfactual reasoning for graph-based intrusion detection, while realistic
remediation and analyst utility remain separate validation requirements.

## Quick Start

Tested with Python 3.11, PyTorch 2.2.2 + CUDA 12.1, Ubuntu 22.04 and one NVIDIA RTX 5880 Ada.

```bash
git clone https://github.com/uet-dis/FlowSCE.git && cd FlowSCE
python -m venv .venv && source .venv/bin/activate
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt     # pinned versions are required: the registered split depends on them

bash run_all.sh                     # all experiments (finished steps are skipped)
python scripts/make_paper_tables.py # paper tables -> results/paper/, from stored ledgers
python scripts/audit.py             # consistency checks of all results
```

`data/` contains the three dataset subsets with provenance and checksums. `scripts/prepare_data.py` rebuilds them from the
official archives:

- [CIC-IDS2018 (NetFlow v3)](https://espace.library.uq.edu.au/view/UQ:ece9b83)
- [DoHBrw-2020](https://www.unb.ca/cic/datasets/dohbrw-2020.html)
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

In the code, `explain_plus` denotes complete FlowSCE, i.e. search followed by refinement.

## Main Results

### Detection

Test macro-F1 (%), mean ± SD over five seeds. The explained detector is the model being explained, not a contribution.

| Model | CIC-IDS2018 | DoHBrw-2020 | UNSW-NB15 |
|---|---|---|---|
| Explained detector | 99.99 ± 0.01 | 99.90 ± 0.06 | 99.26 ± 0.02 |
| Full context | 99.93 ± 0.04 | 98.70 ± 1.99 | 97.99 ± 2.17 |
| Edge only (no host context) | 99.91 ± 0.07 | 99.35 ± 0.09 | 99.26 ± 0.01 |
| E-GraphSAGE | 99.99 ± 0.01 | 99.98 ± 0.05 | 96.54 ± 0.24 |

### Counterfactual explanations

Each dataset has 500 registered alerts (five detector seeds × 100). All methods query the same reconstructed oracle and
validator. Tuning effort and backward passes are not equalized. Values are given as CIC-IDS2018 / DoHBrw-2020 / UNSW-NB15.

| Method | CSR | Changed features (mean) | L1 (median) | Oracle calls (median) |
|---|---|---|---|---|
| CANDICE (tuned λ) | 0.908 / 0.866 / 0.986 | 17.46 / 19.82 / 13.28 | 14.84 / 26.85 / 16.84 | 109 / 193 / 152 |
| DiCE (dice-ml) | 1.000 / 0.030 / 0.398 | 16.38 / 18.27 / 12.98 | 22.20 / 19.38 / 30.35 | 4045 / 4045 / 4045 |
| FlowSCE search + LBS-adapted | 1.000 / 1.000 / 0.998 | 1.05 / 1.37 / 1.02 | 8.09 / 21.09 / 8.24 | +121 / +149 / +115 |
| **FlowSCE** (search + refinement) | 1.000 / 1.000 / 0.998 | 1.07 / 1.29 / 1.00 | 6.04 / 17.57 / 6.97 | 1124 / 1564 / 843 |

The LBS-adapted row starts from the same FlowSCE search seeds; "+" denotes calls added on top of search.

- **Stale context.** Freezing host summaries yields spurious counterfactuals in 1 of 1,499 nominal successes on the
  primary snapshots. Under short-window policies with strong coupling (> 0.5), this rises to 13.8%.
- **Sparsity vs. cost.** FlowSCE and LBS-adapted both reach about one changed feature, and LBS-adapted needs roughly ten
  times fewer oracle calls.
- **Limitations.** Many changed values lie outside the 1st–99th training percentiles. The counterfactuals are verified
  model-input contrasts, not realizable traffic.

Full tables are in `results/SUMMARY.md` and `results/paper/`.

## Citation

Please cite the paper if you use this code; the BibTeX entry will be added on publication.

**License:** MIT | **Contact:** [Issues](https://github.com/uet-dis/FlowSCE/issues)
