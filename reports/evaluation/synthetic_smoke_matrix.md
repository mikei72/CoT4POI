# Synthetic CPU smoke matrix

This is a deterministic CI/demo fixture result, not a research benchmark. The eight labeled
examples contain no real user identifiers or coordinates. All runs are tagged
`production_current` and `rerun` in local MLflow.

| Variant | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@10 | Candidate R@100 | Validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 global | 0.500 | 1.000 | 1.000 | 0.629 | 0.719 | 1.000 | 1.000 |
| B1 + time | 0.500 | 1.000 | 1.000 | 0.629 | 0.719 | 1.000 | 1.000 |
| B2 + transition | 0.625 | 1.000 | 1.000 | 0.744 | 0.806 | 1.000 | 1.000 |
| B3 + history/category | 0.375 | 1.000 | 1.000 | 0.592 | 0.692 | 1.000 | 1.000 |
| B3 no time | 0.250 | 1.000 | 1.000 | 0.529 | 0.646 | 1.000 | 1.000 |
| B3 no transition | 0.000 | 1.000 | 1.000 | 0.331 | 0.496 | 1.000 | 1.000 |
| B3 no history | 0.625 | 1.000 | 1.000 | 0.744 | 0.806 | 1.000 | 1.000 |

- Data manifest SHA-256: `c59fe45b070a504880df6791388f0232fd1b260313c90244fa5caf76534e332e`
- Full aggregate values and per-variant core hashes:
  [`synthetic_smoke_matrix.json`](synthetic_smoke_matrix.json)
- The tiny fixture is intended to prove contracts, lineage, and system closure; differences
  between variants are not statistically meaningful.
