import json
import pandas as pd
from fredapi import Fred


class FredClient:
    def __init__(self, config_path="config/config.json"):
        with open(config_path) as f:
            config = json.load(f)

        self.api = config["fred"]
        self.fred = Fred(api_key=self.api["app_token"])

    def fetch_series(self, series_id):
        rows = self.fred.get_series(series_id)
        print(f"[FredClient] Retrieved {len(rows)} rows for {series_id}")
        return rows

    def fetch_vix(self):
        series = self.fetch_series("VIXCLS")
        if series is None:
            return pd.DataFrame()
        return pd.DataFrame({"date": series.index, "vix": series.values})

    def fetch_usd_index(self):
        series = self.fetch_series("DTWEXBGS")
        if series is None:
            return pd.DataFrame()
        return pd.DataFrame({"date": series.index, "usd_index": series.values})

    def fetch_usd_chf(self):
        series = self.fetch_series("DEXSZUS")
        if series is None:
            return pd.DataFrame()
        return pd.DataFrame({"date": series.index, "usd_chf": series.values})
