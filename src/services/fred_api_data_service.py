import pandas as pd
import numpy as np

from src.clients.fredapi_client import FredClient


class FredMacroService:

    def __init__(self):
        self.client = FredClient()

    def load_dataframes(self):
        vix_df = self.client.fetch_vix()
        usd_index_df = self.client.fetch_usd_index()
        usd_chf_df = self.client.fetch_usd_chf()
        return vix_df, usd_index_df, usd_chf_df

    def merge_and_filter(self, vix_df: pd.DataFrame, usd_index_df: pd.DataFrame, usd_chf_df: pd.DataFrame) -> pd.DataFrame:
        df = (
            vix_df.merge(usd_index_df, on="date", how="outer")
                  .merge(usd_chf_df, on="date", how="outer")
        )

        # date -> datetime (UTC)
        df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
        df = df.sort_values("date")

        # numeric coercion for all value columns
        exclude = ["date"]
        num_cols = [c for c in df.columns if c not in exclude]

        df[num_cols] = (
            df[num_cols]
              .replace({"": np.nan, "NaN": np.nan, None: np.nan})
              .apply(pd.to_numeric, errors="coerce")
        )

        return df

    def load_dataframe(self) -> pd.DataFrame:
        vix_df, usd_index_df, usd_chf_df = self.load_dataframes()
        return self.merge_and_filter(vix_df, usd_index_df, usd_chf_df)
