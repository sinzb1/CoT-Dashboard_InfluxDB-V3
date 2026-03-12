import json
import pandas as pd
from fredapi import Fred
from datetime import date


class FredClient:
    def __init__(self, config_path="config/config.json"):
        with open(config_path) as f:
            config = json.load(f)

        self.api = config["fred"]
        self.years_back = config.get("pipeline", {}).get("years_back", 4)
        self.fred = Fred(api_key=self.api["app_token"])

    def _default_date_range(self):
        end = date.today()
        start = date(end.year - self.years_back, end.month, end.day)
        return start, end

    def fetch_series(self, series_id, observation_start=None, observation_end=None):
        if observation_start is None or observation_end is None:
            observation_start, observation_end = self._default_date_range()
        rows = self.fred.get_series(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        print(f"[FredClient] Retrieved {len(rows)} rows for {series_id} ({observation_start} to {observation_end})")
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
