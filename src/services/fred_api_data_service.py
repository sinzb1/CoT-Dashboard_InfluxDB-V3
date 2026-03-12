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

    def align_to_cot_dates(self, fred_df: pd.DataFrame, cot_dates: pd.Series) -> pd.DataFrame:
        """Align FRED daily data to CoT weekly dates (Tuesday).

        Strategy:
        1. If concrete CoT dates are provided, use them as target dates.
           For each CoT date, pick the closest preceding FRED observation
           (same day or up to 4 business days back) via merge_asof.
        2. If no CoT dates are available, fall back to keeping only
           Tuesday observations from the FRED data.
        """
        if cot_dates is not None and len(cot_dates) > 0:
            # Normalise CoT dates to UTC datetime
            cot_dt = pd.to_datetime(cot_dates, utc=True).drop_duplicates().sort_values().reset_index(drop=True)
            cot_ref = pd.DataFrame({"cot_date": cot_dt})

            fred_sorted = fred_df.sort_values("date").reset_index(drop=True)

            # merge_asof: for each CoT date, find the last FRED observation
            # within a tolerance of 4 days (covers weekends / holidays)
            merged = pd.merge_asof(
                cot_ref,
                fred_sorted,
                left_on="cot_date",
                right_on="date",
                direction="backward",
                tolerance=pd.Timedelta(days=4),
            )

            # Replace the original FRED date with the CoT reference date
            merged["date"] = merged["cot_date"]
            merged = merged.drop(columns=["cot_date"])

            # Drop rows where no FRED match was found
            value_cols = [c for c in merged.columns if c != "date"]
            merged = merged.dropna(subset=value_cols, how="all")

            print(f"[FredMacroService] Aligned {len(merged)} FRED data points to CoT dates")
            return merged.reset_index(drop=True)
        else:
            # Fallback: keep only Tuesday observations
            tuesday_mask = fred_df["date"].dt.dayofweek == 1  # Monday=0, Tuesday=1
            result = fred_df[tuesday_mask].copy().reset_index(drop=True)
            print(f"[FredMacroService] Filtered to {len(result)} Tuesday FRED data points (fallback)")
            return result

    def load_dataframe(self, cot_dates: pd.Series = None) -> pd.DataFrame:
        """Load and merge FRED data, optionally aligned to CoT dates."""
        vix_df, usd_index_df, usd_chf_df = self.load_dataframes()
        merged = self.merge_and_filter(vix_df, usd_index_df, usd_chf_df)
        return self.align_to_cot_dates(merged, cot_dates)
