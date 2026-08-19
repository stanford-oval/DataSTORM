# Query consistency ablation

Measures whether the SQL a report cites is internally consistent — whether
repeated or related queries agree on actors, geography, time windows, event
taxonomy, counting, and population scope. A report is scored only if it cites at
least two SQL queries (`applicable`); the score is the mean agreement across the
six dimensions.

## Results

Two arms, both covering the same 10 topics (1, 2, 4, 8, 9, 10, 11, 12, 17, 18):

| arm | n | mean overall |
|---|---|---|
| `results/baseline/` — query consistency on | 10 | **0.5492** |
| `results/no_qc/` — query consistency off | 10 | 0.4062 |

Removing query consistency costs **0.143** overall and lowers the score on 7 of
the 10 topics. This is Table 4b of the paper.

Six of the ten no-QC reports (topics 2, 4, 8, 10, 11, 12) did not produce usable
cited SQL on the first pass and were re-summarised; each result file's
`report_dir` records which run it was scored from.

## Regenerating

```bash
python run_query_consistency.py --set baseline --ids 1 2 4 8 9 10 11 12 17 18
python run_query_consistency.py --set no_qc    --ids 1 2 4 8 9 10 11 12 17 18
```

Each run writes per-topic scores and a `results/<set>/_summary.json` aggregating
the ids it was given.
