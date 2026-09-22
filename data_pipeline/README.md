# Module 1 — Data Pipeline

Run:

```bash
python pipeline.py
```

The script discovers book categories from Books to Scrape, scrapes at least three categories and continues until there are at least 60 rows.

Cleaning decisions:

- `price_gbp`: numeric extraction; unexpected numeric parsing becomes `NaN` and is median-imputed.
- `rating`: star text is mapped One–Five to 1–5; unexpected values are median-imputed.
- `in_stock`: derived from the presence of `In stock`.
- Rows missing essential `title` or `category` are dropped.
- `price_inr = price_gbp * 105.50`.

Outputs:

- `books.db`
- `query_outputs.txt`

The SQLite schema has `categories(category_id)` as the primary key and `books(category_id)` as a foreign key.
