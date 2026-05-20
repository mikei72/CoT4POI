# CoT Ablation Configs

Each dataset keeps one config file for the CoT-stage ablations.

Planned usage:

- `macro_ablation`: `full`, `w_o_td`, `w_o_preference`, optional `history_only`
- `fine_ablation`: `full`, `w_o_td`, `w_o_preference`, `w_o_macro`, optional `history_only`

The scripts under `ablation_study/cot_ablation/` should read these configs only.
