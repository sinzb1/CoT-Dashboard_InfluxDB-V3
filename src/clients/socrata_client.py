import json
from sodapy import Socrata
from datetime import date


class SocrataClient:
    def __init__(self, config_path="config/config.json"):
        with open(config_path) as f:
            config = json.load(f)

        self.api = config["socrata"]
        self.client = Socrata(self.api["domain"], self.api["app_token"], timeout=600)
        self.limit = self.api["limit"]
        self.max_rows = self.api.get("max_rows", None) # remove property socrata.max_rows for full data fetching
        
        if self.max_rows != None:
            print(f"[SocrataClient] max_rows set to {self.max_rows}. Limiting dataset")

    def get_total_rows(self, dataset_id, filter):
        result = self.client.get(dataset_id, where=filter, select="count(*)")
        try:
            return int(result[0]["count"])
        except Exception:
            return 0

    def fetch_all_rows(self, dataset_id, filter):
        total_rows = self.max_rows or self.get_total_rows(dataset_id, filter)
        rows = []
        offset = 0

        print(f"[SocrataClient] Total rows to fetch: {total_rows} for {dataset_id}")

        while offset < total_rows:
            batch = self.client.get(dataset_id, where=filter , limit=self.limit, offset=offset)

            if not batch:
                break

            rows.extend(batch)

            print(
                f"[SocrataClient] DatasetId={dataset_id} Offset={offset} | Loaded={len(rows)} / {total_rows} "
                f"({round(len(rows) / total_rows * 100, 2)}%)"
            )

            offset += self.limit

        print(f"[SocrataClient] Retrieved {len(rows)} rows for {dataset_id}")
        return rows

    def get_traders_categories(self):
        markets = [
            "GOLD - COMMODITY EXCHANGE INC.",
            "SILVER - COMMODITY EXCHANGE INC.",
            "PLATINUM - NEW YORK MERCANTILE EXCHANGE",
            "PALLADIUM - NEW YORK MERCANTILE EXCHANGE",
            "COPPER- #1 - COMMODITY EXCHANGE INC.",
        ]

        end = date.today()
        start = date(end.year - 10, end.month, end.day)

        escaped = [v.replace("'", "''") for v in markets]
        markets_where = "market_and_exchange_names in ({})".format(
            ",".join(f"'{v}'" for v in escaped)
        )

        date_where = (
            f"report_date_as_yyyy_mm_dd >= '{start.isoformat()}T00:00:00.000' "
            f"AND report_date_as_yyyy_mm_dd <= '{end.isoformat()}T23:59:59.999'"
        )

        where = f"({markets_where}) AND ({date_where})"

        return self.fetch_all_rows("72hh-3qpy", where)
