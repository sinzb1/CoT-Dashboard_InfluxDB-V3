import pandas as pd
from influxdb_client_3 import InfluxDBClient3, Point
import os

from src.clients.fredapi_client import FredClient
from src.services.fred_api_data_service import FredMacroService
from src.services.trades_category_service import TradesCategoryService

# Initialize the InfluxDB v3 client
token = "apiv3_m8zHCYoKyZwSHfrt4oPUMMMDCGD4XZMS6KEV2C9SMchecjhVig4y_27rcHE58uiSSqCjBJby95dsaSNtMYnscA"
database = "CoT-Data"
host = "http://localhost:8181"  # InfluxDB v3 Core default port

# v3: Simplified client initialization (no org parameter)
client = InfluxDBClient3(host=host, token=token, database=database)

# Get data from Socrata
service = TradesCategoryService()
tc_df = service.load_dataframe()
tc_df = service.filter_and_rename(tc_df)

print(f"Writing {len(tc_df)} CoT data points to InfluxDB v3...")

# Iterate through the DataFrame and write data points to InfluxDB
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
            .time(pd.to_datetime(row['Date'], format='%y%m%d'))  # No WritePrecision needed in v3
        
        # v3: Direct write using client.write() (no separate write_api)
        client.write(record=point)
        
    except Exception as e:
        print(f"Error writing CoT data point at index {index}: {e}")
        continue

print("CoT data write completed.")

# Get and write FRED macro data
service = FredMacroService()
fred_df = service.load_dataframe()

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

        # only write if at least one field exists
        if len(p._fields) > 0:
            points.append(p)
            
    except Exception as e:
        print(f"Error preparing macro data point at index {index}: {e}")
        continue

# Batch write all macro points
if points:
    try:
        client.write(record=points)
        print(f"Successfully wrote {len(points)} macro data points.")
    except Exception as e:
        print(f"Error writing macro data batch: {e}")

# v3: Proper cleanup using close()
client.close()
print("InfluxDB v3 client closed. Migration write complete!")
