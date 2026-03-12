import pandas as pd
import numpy as np

from src.clients.yfinance_client import YFinanceClient


class FuturesPriceService:

    def __init__(self):
        self.client = YFinanceClient()

    def load_dataframe(self) -> pd.DataFrame:
        """Load daily close prices for all commodity futures."""
        df = self.client.fetch_close_prices()

        if df.empty or "date" not in df.columns:
            return pd.DataFrame(columns=["date"])

        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)

        # Coerce value columns to numeric
        value_cols = [c for c in df.columns if c != "date"]
        df[value_cols] = (
            df[value_cols]
            .replace({"": np.nan, "NaN": np.nan, None: np.nan})
            .apply(pd.to_numeric, errors="coerce")
        )
        return df

    def align_to_cot_dates(self, prices_df: pd.DataFrame, cot_dates: pd.Series) -> pd.DataFrame:
        """Align daily futures prices to CoT weekly dates (Tuesday).

        Strategy (identical to FredMacroService):
        1. If concrete CoT dates are provided, use merge_asof to pick
           the closest preceding price (tolerance 4 days).
        2. Fallback: keep only Tuesday observations.
        """
        if prices_df.empty:
            return prices_df

        if cot_dates is not None and len(cot_dates) > 0:
            cot_dt = (
                pd.to_datetime(cot_dates, utc=True)
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )
            cot_ref = pd.DataFrame({"cot_date": cot_dt})

            prices_sorted = prices_df.sort_values("date").reset_index(drop=True)

            merged = pd.merge_asof(
                cot_ref,
                prices_sorted,
                left_on="cot_date",
                right_on="date",
                direction="backward",
                tolerance=pd.Timedelta(days=4),
            )

            merged["date"] = merged["cot_date"]
            merged = merged.drop(columns=["cot_date"])

            value_cols = [c for c in merged.columns if c != "date"]
            merged = merged.dropna(subset=value_cols, how="all")

            print(f"[FuturesPriceService] Aligned {len(merged)} price points to CoT dates")
            return merged.reset_index(drop=True)
        else:
            tuesday_mask = prices_df["date"].dt.dayofweek == 1
            result = prices_df[tuesday_mask].copy().reset_index(drop=True)
            print(f"[FuturesPriceService] Filtered to {len(result)} Tuesday price points (fallback)")
            return result

    def load_aligned(self, cot_dates: pd.Series = None) -> pd.DataFrame:
        """Load futures prices and align them to CoT dates."""
        df = self.load_dataframe()
        return self.align_to_cot_dates(df, cot_dates)
