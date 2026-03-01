# Sample Datasets

These are **synthetic sample datasets** generated for use with FMTools's examples and use cases. They do not contain real data.

| File | Schema | Used By | Description |
|------|--------|---------|-------------|
| `server_logs.csv` | `log_id, timestamp, log_message` | `use_cases/01_pipeline_operators/` | Simulated server log entries with various severity levels |
| `medical_notes.csv` | `id, date, raw_note` | `use_cases/02_decorators/` | Fictional doctor dictation notes for triage classification |
| `product_reviews.csv` | `review_id, product, review_text` | `use_cases/03_async_generators/` | Synthetic product review text for sentiment analysis |
| `support_tickets.csv` | `ticket_id, email_subject, email_body` | `use_cases/04_ecosystem_polars/`, `use_cases/05_dspy_optimization/` | Simulated customer support emails |
| `transcript_sample.json` | `FoundationModels.Transcript` JSON shape | `examples/transcript_processing.py` | Synthetic transcript export sample for transcript analytics demo |

The CSV datasets were created with `scripts/generate_datasets.py`.
