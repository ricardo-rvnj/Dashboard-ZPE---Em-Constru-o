import polars as pl

GOV = pl.read_csv("extracted_dfs/comexstat_data.csv")
UN = pl.read_csv("extracted_dfs/comtrade_data.csv")

schema_overrides = {"product_hs92_code": pl.String}
HARVARD = pl.read_csv(
    "extracted_dfs/harvard_data.csv", schema_overrides=schema_overrides
)


print(GOV.columns)
print(UN.columns)
print(HARVARD.columns)
