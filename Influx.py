import json
import pandas as pd
from influxdb_client_3 import InfluxDBClient3, Point
from datetime import date, datetime, timezone

from src.clients.fredapi_client import FredClient
from src.services.fred_api_data_service import FredMacroService
from src.services.trades_category_service import TradesCategoryService
from src.services.futures_price_service import FuturesPriceService

# ── Configuration ────────────────────────────────────────────────────────────
with open("config/config.json") as _f:
    _cfg = json.load(_f)

YEARS_BACK = _cfg.get("pipeline", {}).get("years_back", 4)

token = "apiv3_m8zHCYoKyZwSHfrt4oPUMMMDCGD4XZMS6KEV2C9SMchecjhVig4y_27rcHE58uiSSqCjBJby95dsaSNtMYnscA"
database = "CoT-Data"
host = "http://localhost:8181"  # InfluxDB v3 Core default port

client = InfluxDBClient3(host=host, token=token, database=database)

# ── Helper: targeted delete for a measurement within the 4-year window ───────
def delete_measurement_range(client, measurement: str, start: datetime, end: datetime):
    """Delete data points from *measurement* between *start* and *end*.

    InfluxDB v3 Core supports DELETE via SQL.
    If the engine does not support DELETE, the error is logged but execution
    continues so that the new data can still be written (idempotent upsert
    behaviour of InfluxDB on identical timestamps).
    """
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    delete_sql = (
        f"DELETE FROM \"{measurement}\" "
        f"WHERE time >= '{start_str}' AND time <= '{end_str}'"
    )
    print(f"[Delete] Executing: {delete_sql}")
    try:
        client.query(query=delete_sql, language="sql")
        print(f"[Delete] Successfully deleted {measurement} data from {start_str} to {end_str}")
    except Exception as e:
        print(
            f"[Delete] Could not delete from {measurement} ({e}). "
            "Data will be overwritten by upsert on identical timestamps."
        )

# ── Time window ──────────────────────────────────────────────────────────────
today = date.today()
window_start = datetime(today.year - YEARS_BACK, today.month, today.day, tzinfo=timezone.utc)
window_end = datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc)

# For targeted delete we remove everything from well before the earliest
# possible data up to today so that legacy data (>4 years) is also cleaned:
delete_start = datetime(2000, 1, 1, tzinfo=timezone.utc)

# ── 1. CoT Data ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Pipeline: loading CoT + FRED + Futures data for last {YEARS_BACK} years")
print(f"Window:   {window_start.date()} → {window_end.date()}")
print(f"{'='*60}\n")

service = TradesCategoryService()
tc_df = service.load_dataframe()
tc_df = service.filter_and_rename(tc_df)

print(f"Loaded {len(tc_df)} CoT data points from Socrata.\n")

# Extract unique CoT dates for FRED alignment
cot_dates = tc_df["Date"].drop_duplicates().sort_values()
print(f"Unique CoT report dates: {len(cot_dates)}")

# Targeted delete: remove old CoT data (including legacy >4yr data)
delete_measurement_range(client, "cot_data", delete_start, window_end)

print(f"\nWriting {len(tc_df)} CoT data points to InfluxDB v3...")

cot_points = []

for index, row in tc_df.iterrows():
    try:
        point = Point("cot_data") \
            .tag("market_names", row['Market Names']) \
            .field("Open Interest", float(row['Open Interest'])) \
            .field("Producer/Merchant/Processor/User Long", float(row['Producer/Merchant/Processor/User Long'])) \
            .field("Producer/Merchant/Processor/User Short", float(row['Producer/Merchant/Processor/User Short'])) \
            .field("Swap Dealer Long", float(row['Swap_Dealer_Long'])) \
            .field("Swap Dealer Short", float(row['Swap_Dealer_Short'])) \
            .field("Swap Dealer Spread", float(row['Swap_Dealer_Spread'])) \
            .field("Managed Money Long", float(row['Managed_Money_Long'])) \
            .field("Managed Money Short", float(row['Managed_Money_Short'])) \
            .field("Managed Money Spread", float(row['Managed_Money_Spread'])) \
            .field("Other Reportables Long", float(row['Other_Reportables_Long'])) \
            .field("Other Reportables Short", float(row['Other_Reportables_Short'])) \
            .field("Other Reportables Spread", float(row['Other_Reportables_Spread'])) \
            .field("Total Traders", float(row['Total_Traders'])) \
            .field("Traders Prod/Merc Long", float(row['Traders_Prod_Merc_Long'])) \
            .field("Traders Prod/Merc Short", float(row['Traders_Prod_Merc_Short'])) \
            .field("Traders Swap Long", float(row['Traders_Swap_Long'])) \
            .field("Traders Swap Short", float(row['Traders_Swap_Short'])) \
            .field("Traders Swap Spread", float(row['Traders_Swap_Spread'])) \
            .field("Traders M Money Long", float(row['Traders_M_Money_Long'])) \
            .field("Traders M Money Short", float(row['Traders_M_Money_Short'])) \
            .field("Traders M Money Spread", float(row['Traders_M_Money_Spread'])) \
            .field("Traders Other Rept Long", float(row['Traders_Other_Rept_Long'])) \
            .field("Traders Other Rept Short", float(row['Traders_Other_Rept_Short'])) \
            .field("Traders Other Rept Spread", float(row['Traders_Other_Rept_Spread'])) \
            .time(row['Date'])

        cot_points.append(point)

    except Exception as e:
        print(f"Error preparing CoT data point at index {index}: {e}")
        continue

if cot_points:
    try:
        client.write(record=cot_points)
        print(f"Successfully wrote {len(cot_points)} CoT data points.")
    except Exception as e:
        print(f"Error writing CoT data batch: {e}")

print("CoT data write completed.")

# ── 2. FRED Macro Data (aligned to CoT dates) ───────────────────────────────
fred_service = FredMacroService()
fred_df = fred_service.load_dataframe(cot_dates=cot_dates)

print(f"\n{len(fred_df)} FRED data points aligned to CoT dates.")

# Targeted delete: remove old macro data (including legacy >4yr data)
delete_measurement_range(client, "macro_by_date", delete_start, window_end)

print(f"Writing {len(fred_df)} macro data points to InfluxDB v3...")

points = []

for index, row in fred_df.iterrows():
    try:
        p = Point("macro_by_date").time(row["date"].to_pydatetime())

        if pd.notna(row.get("vix")):
            p = p.field("vix", float(row["vix"]))
        if pd.notna(row.get("usd_index")):
            p = p.field("usd_index", float(row["usd_index"]))
        if pd.notna(row.get("usd_chf")):
            p = p.field("usd_chf", float(row["usd_chf"]))

        if len(p._fields) > 0:
            points.append(p)

    except Exception as e:
        print(f"Error preparing macro data point at index {index}: {e}")
        continue

if points:
    try:
        client.write(record=points)
        print(f"Successfully wrote {len(points)} macro data points.")
    except Exception as e:
        print(f"Error writing macro data batch: {e}")

# ── 3. Futures Price Data (aligned to CoT dates) ────────────────────────────
futures_service = FuturesPriceService()
futures_df = futures_service.load_aligned(cot_dates=cot_dates)

print(f"\n{len(futures_df)} Futures price points aligned to CoT dates.")

# Targeted delete: remove old futures price data (including legacy >4yr data)
delete_measurement_range(client, "futures_prices", delete_start, window_end)

print(f"Writing {len(futures_df)} futures price points to InfluxDB v3...")

futures_points = []

for index, row in futures_df.iterrows():
    try:
        p = Point("futures_prices").time(row["date"].to_pydatetime())

        if pd.notna(row.get("gold")):
            p = p.field("gold_close", float(row["gold"]))
        if pd.notna(row.get("silver")):
            p = p.field("silver_close", float(row["silver"]))
        if pd.notna(row.get("copper")):
            p = p.field("copper_close", float(row["copper"]))
        if pd.notna(row.get("platinum")):
            p = p.field("platinum_close", float(row["platinum"]))
        if pd.notna(row.get("palladium")):
            p = p.field("palladium_close", float(row["palladium"]))

        if len(p._fields) > 0:
            futures_points.append(p)

    except Exception as e:
        print(f"Error preparing futures price point at index {index}: {e}")
        continue

if futures_points:
    try:
        client.write(record=futures_points)
        print(f"Successfully wrote {len(futures_points)} futures price points.")
    except Exception as e:
        print(f"Error writing futures price batch: {e}")

# ── Cleanup ──────────────────────────────────────────────────────────────────
client.close()
print("\nInfluxDB v3 client closed. Pipeline complete!")
