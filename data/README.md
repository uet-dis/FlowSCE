# Data

The three experiment datasets are fixed subsets of the official releases; the complete DoHBrw-2020 layer-2 capture, used only for the coupling study, is built by `scripts/prepare_doh_full.py` and not stored in the repository. The subsets are built row by row by `scripts/prepare_data.py` from the official archives, using the row lists in `source_rows/` (`row_id` → source file and 0-based data row). `SHA256SUMS` holds the SHA-256 of a canonical CSV rendering of each file (independent of the parquet writer), which `prepare_data.py` checks after rebuilding.

| File | Release | Rows | Classes |
|---|---|---|---|
| `nf_cicids2018_v3_4class_60k.parquet` | NF-CICIDS2018-v3 (UQ eSpace, UQ:ece9b83) | 60,000 | Benign 15,000; FTP-BruteForce 15,000; DoS_attacks-Hulk 15,000; Bot 15,000 |
| `cira_cic_dohbrw2020_l2_49807.parquet` | CIRA-CIC-DoHBrw-2020, layer 2 (UNB CIC) | 49,807 | Malicious 30,000; Benign 19,807 |
| `unsw_nb15_60k.parquet` | UNSW-NB15 (UNSW Canberra), UNSW-NB15_1..4.csv | 60,000 | Normal 30,000; Generic 20,087; Exploits 4,187; Fuzzers 2,229; DoS 1,507; Reconnaissance 1,318; Analysis 256; Backdoor 182; Shellcode 157; Backdoors 52; Worms 25 |
| `cira_cic_dohbrw2020_l2_full.parquet` | CIRA-CIC-DoHBrw-2020, layer 2, complete capture (`scripts/prepare_doh_full.py`) | 269,643 | Malicious 249,836; Benign 19,807 |

## Sampling

- **cic** — Benign: 15,000 of 17,514,626; FTP-BruteForce: 15,000 of 386,720; Bot: 15,000 of 207,703; DoS_attacks-Hulk: 15,000 of 100,076. Source files: `NF-CICIDS2018-v3.csv` (20,115,529 rows, SHA-256 `242a6971cc801eae…`).
- **doh** — Malicious: 30,000 of 249,836; Benign: 19,807 of 19,807. Source files: `l2-benign.csv` (19,807 rows, SHA-256 `f5a96b741f1a2eee…`), `l2-malicious.csv` (249,836 rows, SHA-256 `fcee1ecf8c73ae14…`).
- **unsw** — Normal: 30,000 of 2,218,764; Generic: 20,087 of 215,481; Exploits: 4,187 of 44,525; Fuzzers: 2,229 of 24,246; DoS: 1,507 of 16,353; Reconnaissance: 1,318 of 13,987; Analysis: 256 of 2,677; Backdoor: 182 of 1,795; Shellcode: 157 of 1,511; Backdoors: 52 of 534; Worms: 25 of 174. Source files: `UNSW-NB15_1.csv` (700,001 rows, SHA-256 `7d851bbeabd27894…`), `UNSW-NB15_2.csv` (700,001 rows, SHA-256 `6130ad02873cc606…`), `UNSW-NB15_3.csv` (700,001 rows, SHA-256 `ae990a96c3dfcd42…`), `UNSW-NB15_4.csv` (440,044 rows, SHA-256 `cdf563692d51d405…`).

## Columns

Original columns keep their release names and values (text stripped; integer columns as nullable integers; hexadecimal ports parsed to integers; DoH `TimeStamp` as datetime). Added columns: `row_id`, `dataset`, `label` (0 benign / 1 attack), `attack_type` (harmonised class name; UNSW-NB15 'Backdoors' → 'Backdoor', benign → 'Benign'), `source_file`, `source_row`. Splits stratify on the original class columns (`Attack`, `Label`, `attack_cat`).

## Notes

- The UNSW-NB15 and NF-CICIDS2018-v3 releases contain exact duplicate rows; `source_row` lists the first unused identical row.
- Feature dictionaries shipped with the releases (`NetFlow_v3_Features.csv`, `NUSW-NB15_features.csv`) and the byte–duration–rate relations tested on the subsets are stored in `provenance/`; only the DoHBrw-2020 rate identities hold exactly.
- Licensing and citation requirements of the original datasets apply to these subsets.
