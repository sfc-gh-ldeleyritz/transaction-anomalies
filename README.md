# Transaction anomalies

Streamlit-in-Snowflake app from `SWT_DEMO.FRAUD.TRANSACTION_ANOMALIES`.

Scores card transactions in `SWT_DEMO.FRAUD.TRANSACTIONS` with four Snowflake-side rules:

- amount spike: z-score of `LN(amount)` against the cardholder history
- unusual geography: spend outside the cardholder home region
- velocity burst: 6 or more authorisations in the same hour
- impossible travel: two distant countries inside 60 minutes

## Run in Snowflake

Warehouse: `COMPUTE_WH`  
Runtime: Streamlit container (`SYSTEM$ST_CONTAINER_RUNTIME_PY3_11`)  
Main file: `app.py`

```sql
CREATE STREAMLIT SWT_DEMO.FRAUD.TRANSACTION_ANOMALIES
  FROM '.'
  MAIN_FILE = 'app.py'
  QUERY_WAREHOUSE = COMPUTE_WH;
```

Or `snow streamlit deploy` from this directory after setting `snowflake.yml`.

## Dependencies

See `pyproject.toml`. `plotly` needs the account PyPI external access integration (`PYPI_ACCESS_INTEGRATION`).
