# Clinical Data Analysis

A reproducible Python portfolio project for data-quality checks, descriptive analysis, laboratory-result summaries and exploratory statistical methods.

> **Data note:** All included rows are synthetic demonstration data. The outcome label is simulated and not verified against longitudinal follow-up. Laboratory bounds are illustrative. This is not clinical evidence, diagnostic guidance or a validated healthcare dataset.

## What it demonstrates

- Schema, uniqueness and referential-integrity checks
- Checks for missing identifiers, invalid flags and inconsistent example lab bounds
- Department and month-level summaries
- Exploratory comparison using Welch's t-test
- CSV/JSON outputs, plots and unit tests

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/analyze_clinical_data.py
python -m unittest discover -s tests -v
```

Generated outputs are written to `outputs/` and are not committed.

## Documentation

- [Analysis plan](docs/analysis-plan.md)
- [Data dictionary](docs/data-dictionary.md)
- [Data validity audit](docs/clinical-evidence-and-data-audit.md)

## Limitations

This is a demonstration of reproducible analysis, not a clinical study. Statistical tests on synthetic records cannot establish real clinical associations, causation or generalizability. Before using a public dataset, document its source, licence, cohort definitions and limitations; protect privacy and seek domain review.
