from typing import TypedDict, Any, Dict
import comtradeapicall as comtrade
import polars as pd
import time
import os


class _comtrade_filters(TypedDict, total=False):
    typeCode: str
    freqCode: str
    clCode: str
    period: str
    reporterCode: str | None
    cmdCode: str
    flowCode: str
    partnerCode: str
    partner2Code: str
    customsCode: str
    motCode: str | None
    maxRecords: int | None
    format_output: str | None
    aggregateBy: str | None
    breakdownMode: str | None
    countOnly: bool
    includeDesc: bool


class Comtrade:
    def __init__(self, comtrade_key=None) -> None:
        self.comtrade_key = self._get_key(comtrade_key)

    def _get_key(self, key=None) -> str:
        if key is None:
            key = os.getenv("COMTRADE_API_KEY")
            if not key:
                raise ValueError(
                    "No subscription key provided and 'COMTRADE_API_KEY' not found in environment."
                )
        return key

    def query_data(
        self, save_csv=False, max_records=None, **filters: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Fetch trade data from the UN Comtrade API using comtradeapicall library.

        Parameters
        ----------
        subscription_key : str, optional
            Your UN Comtrade API key. If None, will read from environment variable 'COMTRADE_API_KEY'.
        save_csv : bool, optional
            Whether to save results as a CSV file (default False).
        max_records : int, optional
            Maximum number of records to fetch (default 100).
        filters : dict
            Any additional filters supported by the API.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with the trade data.
        """
        if self.comtrade_key is None:
            self.comtrade_key = os.getenv("COMTRADE_API_KEY")
            if not self.comtrade_key:
                raise ValueError(
                    "No subscription key provided and 'COMTRADE_API_KEY' not found in environment."
                )

        # Ensure default filters
        default_filters = _comtrade_filters(
            typeCode="C",
            freqCode="A",
            clCode="HS",
            period="2023",
            reporterCode=None,
            cmdCode="AG4",
            flowCode="X",
            partnerCode="76",
            partner2Code="0",
            customsCode="C00",
            motCode=None,
            maxRecords=None,
            format_output=None,
            aggregateBy=None,
            breakdownMode="plus",
            countOnly=False,
            includeDesc=True,
        )

        final_filters = {**default_filters, **filters}

        try:
            print("🔎 Fetching data from UN Comtrade API...")
            df = comtrade.getFinalData(
                subscription_key=self.comtrade_key, **final_filters
            )

            if df is None or df.empty:
                print("⚠️ No data found for the given filters.")
                return pd.DataFrame()

            print(f"✅ Retrieved {len(df)} rows.")

            if save_csv:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                file_name = f"comtrade_{timestamp}.csv"
                df.to_csv(file_name, index=False, encoding="latin1")
                print(f"💾 Data saved to '{file_name}'")

            return df

        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()
