import polars as pl
import os

UN = pl.read_csv("resources/comtrade_data.csv")

if __name__ == "__main__":
    print(UN.schema.keys())
    print(UN.select("classificationSearchCode"))
