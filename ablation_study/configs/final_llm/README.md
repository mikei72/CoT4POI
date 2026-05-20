# Final LLM Ablation Configs

Keep the costly final-stage ablations isolated from the main training pipeline.

Recommended setup:

- run the full final-stage ablation on `nyc`
- keep `tky` and `ca` as placeholders unless extra budget is available
- use end-to-end chain ablation as the main experiment
- keep input masking as optional analysis only
