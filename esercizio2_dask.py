import numpy as np
import pandas as pd
import dask.dataframe as dd

rng = np.random.default_rng(42)

n = 100_000

df = pd.DataFrame({
    "timestamp": pd.date_range("2025-01-01", periods=n, freq="min"),
    "value": rng.normal(100, 10, n),
    "volume": rng.integers(100, 1000, n)
})

df.to_csv("transazioni_demo.csv", index=False)