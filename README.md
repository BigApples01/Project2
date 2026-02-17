# Project2: Exit Survey Ranking Analysis

## Research question
This project answers the question: **"Rank order the programs or courses based on student ratings or preferences for that year."**
It reads the anonymized one-year exit survey dataset in `data/`, computes deterministic item rankings from rating/preference responses, and produces tabular and visual outputs.

## Run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Preview the dataset structure in logs:
   ```bash
   python src/preview.py
   ```
3. Run the rank-order analysis:
   ```bash
   python src/rank_order.py
   ```

## Outputs
Generated outputs are written to `outputs/`:
- `outputs/rank_order.csv`: ranked items with `item`, `mean_score`, `n`, `rank`
- `outputs/rank_order.png`: horizontal bar chart of top ranked items

A GitHub Actions workflow at `.github/workflows/run_analysis.yml` runs the same steps on pushes to `main` and via manual dispatch, then uploads `outputs/` as an artifact named `outputs`.
