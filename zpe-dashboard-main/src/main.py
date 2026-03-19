import polars as pl
import os
from dotenv import load_dotenv
load_dotenv()


def comtrade():
    from comtrade import Comtrade

    comtrade = Comtrade()
    comtrade_df = pl.DataFrame(
        comtrade.query_data(
            partnerCode="76",
            typeCode="C",
            freqCode="A",
            clCode="HS",
        )
    )
    comtrade_df = comtrade_df.filter(pl.col("classificationCode") == "H4")

    return comtrade_df


def harvard():
    from dataverse import HarvardDataverse

    TOKEN = os.getenv("HARVARD_API_KEY")
    DOI: str = "doi:10.7910/DVN/T4CHWJ"

    dataverse = HarvardDataverse(api_token=TOKEN)

    schema_override = {"product_hs92_code": pl.Utf8}

    harvard_df = dataverse.import_df(
        doi=DOI,
        target_filename="hs92_country_product_year_4.csv",
        polars_reader_options={"schema_overrides": schema_override},
    )
    harvard_df = harvard_df.filter(pl.col("year") == pl.col("year").max())

    return harvard_df


def comexstat():
    from comexstat import Comexstat

    comex = ComexStat()
    comexstat_df = comex.query_comexstat_data(
        flow="export",
        period_from="2023-01",
        period_to="2023-12",
        # filters=[{"filter": "state", "values": [23]}],
        metrics=["metricFOB"],
        details=["state", "heading"],
    )
    return comexstat_df


if __name__ == "__main__":
    print("Comexstat schema:")
    # comexstat_df = comexstat()
    # print(comexstat_df.schema)
    # comexstat_df.write_csv("resources/comexstat_data.csv")

    # COMEXSTAT
    comexstat_df = comexstat()
    output_path = os.path.join(BASE_DIR, "resources", "comexstat_data.csv")
    comexstat_df.write_csv(output_path)

    print("\nHarvard schema:")
    # harvard_df = pl.read_csv(
    #     "Dashboard-Base/harvard_data.csv",
    #     schema_overrides={"product_hs92_code": pl.Utf8},
    # )
    # harvard_df = harvard()
    # print(harvard_df.schema)
    # harvard_df.write_csv("resources/harvard_data.csv")
    # HARVARD
    harvard_df = harvard()
    output_path = os.path.join(BASE_DIR, "resources", "harvard_data.csv")
    harvard_df.write_csv(output_path)

    print("\nComtrade schema:")
    # comtrade_df = pl.read_csv("Dashboard-Base/comtrade_data.csv")
    comtrade_df = comtrade()
    print(comtrade_df.schema)

    # MUDEI AQUI
    # comtrade_df.write_csv("resources/comtrade_data.csv")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(BASE_DIR, "resources", "comtrade_data.csv")
    comtrade_df.write_csv(output_path)

# # Convert columns to consistent data types
# # Fix year columns - convert all to Int64
# comexstat_df = comexstat_df.with_columns(
#     pl.col("year").cast(pl.Int64),
#     pl.col("headingCode").cast(pl.Utf8),
#     pl.col("metricFOB").cast(pl.Float64)
# )

# harvard_df = harvard_df.with_columns(
#     pl.col("year").cast(pl.Int64),
#     pl.col("product_hs92_code").cast(pl.Utf8),
#     pl.col("country_iso3_code").cast(pl.Utf8)
# )

# comtrade_df = comtrade_df.with_columns(
#     pl.col("refYear").cast(pl.Int64),
#     pl.col("cmdCode").cast(pl.Utf8),
#     pl.col("reporterISO").cast(pl.Utf8)
# )

# # Filter for valid numeric HS codes (4-digit codes)
# numeric_hs_filter = pl.col("product_hs92_code").str.contains(r"^\d{4}$")
# harvard_numeric = harvard_df.filter(numeric_hs_filter)

# numeric_cmd_filter = pl.col("cmdCode").str.contains(r"^\d{4}$")
# comtrade_numeric = comtrade_df.filter(numeric_cmd_filter)

# # First merge: Harvard and Comtrade
# merged_harvard_comtrade = harvard_numeric.join(
#     comtrade_numeric,
#     left_on=["product_hs92_code", "year", "country_iso3_code"],
#     right_on=["cmdCode", "refYear", "reporterISO"],
#     how="left",
#     suffix="_comtrade"
# )

# # Prepare comexstat data - aggregate to country level
# comexstat_agg = comexstat_df.group_by(["headingCode", "year"]).agg(
#     pl.sum("metricFOB").alias("total_metricFOB"),
#     pl.count().alias("state_records_count")
# )

# # Second merge: Add comexstat data
# final_df = merged_harvard_comtrade.join(
#     comexstat_agg,
#     left_on=["product_hs92_code", "year"],
#     right_on=["headingCode", "year"],
#     how="left",
#     suffix="_comexstat"
# )

# print(f"Final merged dataset shape: {final_df.shape}")
# print("\nFinal columns:")
# print(final_df.columns)

# # Show sample of the merged data
# print("\nSample of merged data:")
# print(final_df.head(5))

# # print (final_df)
# # final_df.write_csv('Dashboard-Base/test_dashboard.csv')

