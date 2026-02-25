import pandas as pd
import numpy as np

from src.clients.socrata_client import SocrataClient
from src.mappings.categories_of_traders_column_map import COLUMN_MAP


class TradesCategoryService:
    def __init__(self):
        self.client = SocrataClient()

    def load_dataframe(self):
        rows = self.client.get_traders_categories()
        df = pd.DataFrame.from_records(rows)
        return df

    def filter_and_rename(self, df):
        filtered_df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

        market_filter = {
            "GOLD - COMMODITY EXCHANGE INC.": "Gold",
            "SILVER - COMMODITY EXCHANGE INC.": "Silver",
            "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "Platinum",
            "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": "Palladium",
            "COPPER- #1 - COMMODITY EXCHANGE INC.": "Copper"
        }

        # Rename column
        filtered_df["Market Names"] = filtered_df["Market Names"].replace(market_filter)

        # convert to date_time
        filtered_df["Date"] = pd.to_datetime(filtered_df["Date"], format="mixed", utc=True)

        # convert data field to numerics
        exclude = ["Market Names", "Date"]
        num_cols = [c for c in filtered_df.columns if c not in exclude]
        filtered_df[num_cols] = (
            filtered_df[num_cols]
            .replace({"": np.nan, "NaN": np.nan, None: np.nan})
            .apply(pd.to_numeric, errors="coerce")
        )

        return filtered_df
