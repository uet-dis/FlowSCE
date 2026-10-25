# Case studies (generated from the query ledgers)
Model-input contrasts in raw feature units (rounded for display), not validated network remediation. Coupled statistics are not checked.

## cic
### (i) Spurious validity (B)
None observed on the full snapshot: every nominal B success was confirmed by reconstruction.
### (ii) Restoration keeps the label but lowers the margin
query 1037 (seed 71): support 17 -> 1, logit margin 4.797 -> 0.316; reconstructed score q=-0.3158 (nominal q=-0.3158)
| feature | unit | raw_original | raw_counterfactual | processed_original | processed_counterfactual | transform |
|---|---|---|---|---|---|---|
| NUM_PKTS_128_TO_256_BYTES | packets | 1 | 2 | 0.4272 | 1.326 | std |
### (iii) FlowSCE explanation
query 11669 (seed 23, true_alert): search output (A0) support 21, L1 19.43, logit margin 3.499 -> FlowSCE support 1, L1 4.33, logit margin 0.817; reconstructed score q=-0.8173; 1121 oracle calls
| feature | unit | raw_original | raw_counterfactual | processed_original | processed_counterfactual | transform |
|---|---|---|---|---|---|---|
| DST_TO_SRC_AVG_THROUGHPUT | bits/s | 2.48e+05 | 0 | 0.6065 | -3.724 | log1p+std |
### (iv) Failure
No failure among the registered alerts or in the time-window sweep.

## doh
### (i) Spurious validity (B)
None observed on the full snapshot: every nominal B success was confirmed by reconstruction.
### (ii) Restoration keeps the label but lowers the margin
query 3146 (seed 37): support 20 -> 1, logit margin 6.442 -> 0.035; reconstructed score q=-0.0346 (nominal q=-0.0346)
| feature | unit | raw_original | raw_counterfactual | processed_original | processed_counterfactual | transform |
|---|---|---|---|---|---|---|
| FlowBytesSent | bytes | 110 | 343 | -1.518 | -1.018 | log1p+std |
### (iii) FlowSCE explanation
query 3053 (seed 23, false_alert): search output (A0) support 20, L1 10.02, logit margin 5.968 -> FlowSCE support 1, L1 0.43, logit margin 0.759; reconstructed score q=-0.7590; 648 oracle calls
| feature | unit | raw_original | raw_counterfactual | processed_original | processed_counterfactual | transform |
|---|---|---|---|---|---|---|
| ResponseTimeTimeMode | unknown | 1.4e-05 | -0.8594 | -0.1382 | -0.5646 | std |
### (iv) Failure
No failure among the registered alerts of the full snapshot. Example from the time-window sweep (Δt = 1 s, detector seed 53, retrained for that policy): query 4348 (true alert), stop=budget_stop, original score q=6.042, coupling ω=0.236; no candidate passed the reconstructed oracle and the validator within the budget (500 updates, 502 forward evaluations). Mutable features of the alert:
| feature | unit | raw_original | raw_counterfactual | processed_original | processed_counterfactual | transform |
|---|---|---|---|---|---|---|
| Duration | seconds | 120.3 | – | 1.388 | – | std |
| FlowBytesSent | bytes | 1.622e+04 | – | 0.6857 | – | log1p+std |
| FlowSentRate | bytes/s | 134.8 | – | -0.1023 | – | std |
| FlowBytesReceived | bytes | 2.753e+04 | – | 0.7418 | – | log1p+std |
| PacketLengthVariance | bytes^2 | 4.752e+04 | – | 0.4342 | – | log1p+std |
| PacketLengthMedian | bytes | 76 | – | -0.6157 | – | log1p+std |
| PacketLengthMode | bytes | 68 | – | -0.08898 | – | log1p+std |
| PacketLengthSkewFromMedian | dimensionless | 1.209 | – | 0.4447 | – | std |
| PacketLengthSkewFromMode | dimensionless | 0.4398 | – | 0.1052 | – | std |
| PacketTimeVariance | unknown | 459 | – | 0.7566 | – | log1p+std |
| PacketTimeMode | unknown | 0 | – | -0.256 | – | log1p+std |
| PacketTimeSkewFromMedian | dimensionless | 0.1632 | – | -0.08877 | – | std |
| PacketTimeSkewFromMode | dimensionless | 1.378 | – | 0.1539 | – | std |
| PacketTimeCoefficientofVariation | dimensionless | 0.7257 | – | -0.4233 | – | std |
| ResponseTimeTimeVariance | unknown | 0.009069 | – | -0.173 | – | std |
| ResponseTimeTimeMean | unknown | 0.04828 | – | -0.2217 | – | std |
| ResponseTimeTimeMode | unknown | 0.02679 | – | -0.1249 | – | std |
| ResponseTimeTimeSkewFromMedian | dimensionless | 0.6801 | – | 0.5052 | – | std |
| ResponseTimeTimeSkewFromMode | dimensionless | 0.2257 | – | 0.04185 | – | std |
| ResponseTimeTimeCoefficientofVariation | dimensionless | 1.972 | – | 0.3639 | – | std |

## unsw
### (i) Spurious validity (B)
query 5634 (row 28159, seed 71): nominal (stale) score q=-0.0002, reconstructed score q=0.0004
| feature | unit | raw_original | raw_counterfactual | processed_original | processed_counterfactual | transform |
|---|---|---|---|---|---|---|
| dur | seconds | 1.031 | 0 | 0.0921 | -0.164 | std |
| Sload | bits/s | 7850 | 4.392e+09 | -1.59 | 1.992 | log1p+std |
| dmeansz | bytes (mean packet size) | 1012 | 1471 | 1.513 | 1.65 | log1p+std |
| tcprtt | unknown | 0 | 1.11 | -0.2641 | 21.18 | std |
### (ii) Restoration keeps the label but lowers the margin
query 11137 (seed 11): support 12 -> 3, logit margin 5.295 -> 0.632; reconstructed score q=-0.6324 (nominal q=-0.6324)
| feature | unit | raw_original | raw_counterfactual | processed_original | processed_counterfactual | transform |
|---|---|---|---|---|---|---|
| sbytes | bytes | 114 | 1039 | -0.9021 | 0.4154 | log1p+std |
| smeansz | bytes (mean packet size) | 57 | 127 | -0.6112 | 0.6969 | log1p+std |
| tcprtt | unknown | 0 | 0.06759 | -0.2641 | 1.042 | std |
### (iii) FlowSCE explanation
query 5524 (seed 11, false_alert): search output (A0) support 15, L1 18.20, logit margin 0.436 -> FlowSCE support 1, L1 12.21, logit margin 1.119; reconstructed score q=-1.1189; 874 oracle calls
| feature | unit | raw_original | raw_counterfactual | processed_original | processed_counterfactual | transform |
|---|---|---|---|---|---|---|
| tcprtt | unknown | 0.1153 | 0.7472 | 1.965 | 14.17 | std |
### (iv) Failure
full snapshot, query 11275 (true_alert), stop=budget_stop, original q=6.822
