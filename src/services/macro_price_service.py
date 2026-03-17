import pandas as pd
import numpy as np

from src.clients.yfinance_client import YFinanceClient


class MacroPriceService:
    """Loads macro factor time series (VIX, USD Index, USD/CHF) via yfinance
    and aligns them to CoT weekly dates — identical logic to FuturesPriceService.
    """

    def __init__(self):
        self.client = YFinanceClient()

    def load_dataframe(self) -> pd.DataFrame:
        """Load daily close prices for all macro tickers."""
        df = self.client.fetch_macro_close_prices()

        if df.empty or "date" not in df.columns:
            return pd.DataFrame(columns=["date"])

        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)

        value_cols = [c for c in df.columns if c != "date"]
        df[value_cols] = (
            df[value_cols]
            .replace({"": np.nan, "NaN": np.nan, None: np.nan})
            .apply(pd.to_numeric, errors="coerce")
        )
        return df

    def align_to_cot_dates(self, macro_df: pd.DataFrame, cot_dates: pd.Series) -> pd.DataFrame:
        """Align daily macro prices to CoT weekly dates (Tuesday).

        Identical strategy to FuturesPriceService.align_to_cot_dates.
        """
        if macro_df.empty:
            return macro_df

        if cot_dates is not None and len(cot_dates) > 0:
            cot_dt = (
                pd.to_datetime(cot_dates, utc=True)
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )
            cot_ref = pd.DataFrame({"cot_date": cot_dt})

            macro_sorted = macro_df.sort_values("date").reset_index(drop=True)

            merged = pd.merge_asof(
                cot_ref,
                macro_sorted,
                left_on="cot_date",
                right_on="date",
                direction="backward",
                tolerance=pd.Timedelta(days=4),
            )

            merged["date"] = merged["cot_date"]
            merged = merged.drop(columns=["cot_date"])

            value_cols = [c for c in merged.columns if c != "date"]
            merged = merged.dropna(subset=value_cols, how="all")

            print(f"[MacroPriceService] Aligned {len(merged)} macro data points to CoT dates")
            return merged.reset_index(drop=True)
        else:
            tuesday_mask = macro_df["date"].dt.dayofweek == 1
            result = macro_df[tuesday_mask].copy().reset_index(drop=True)
            print(f"[MacroPriceService] Filtered to {len(result)} Tuesday macro points (fallback)")
            return result

    def load_aligned(self, cot_dates: pd.Series = None) -> pd.DataFrame:
        """Load macro prices and align them to CoT dates."""
        df = self.load_dataframe()
        return self.align_to_cot_dates(df, cot_dates)
