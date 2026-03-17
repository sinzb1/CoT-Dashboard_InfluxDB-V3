import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
from influxdb_client_3 import InfluxDBClient3
import plotly.graph_objs as go
import webbrowser
from threading import Timer
from datetime import datetime as dt, timedelta
import dash_bootstrap_components as dbc
import numpy as np

# Function to open the web browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8051/")

# Connect to InfluxDB v3
host = "http://localhost:8181"  # InfluxDB v3 Core default port
token = "apiv3_m8zHCYoKyZwSHfrt4oPUMMMDCGD4XZMS6KEV2C9SMchecjhVig4y_27rcHE58uiSSqCjBJby95dsaSNtMYnscA"
database = "CoT-Data"

print("Connecting to InfluxDB v3...")
client = InfluxDBClient3(host=host, token=token, database=database)

# Define SQL query to fetch data (v3 Core uses SQL instead of Flux)
query = """
SELECT *
FROM cot_data
WHERE time >= now() - INTERVAL '4 years'
"""

print("Fetching data from InfluxDB v3...")
# Execute query - v3 returns data as PyArrow Table, convert to pandas DataFrame
table = client.query(query=query, language="sql")

# Convert PyArrow Table to pandas DataFrame
df_pivoted = table.to_pandas()

print(f"Fetched {len(df_pivoted)} rows from InfluxDB v3")

# Fetch futures prices for PPCI (stored by the pipeline in futures_prices measurement)
query_futures = """
SELECT *
FROM futures_prices
WHERE time >= now() - INTERVAL '4 years'
"""
print("Fetching futures prices from InfluxDB v3...")
try:
    table_futures = client.query(query=query_futures, language="sql")
    df_futures_prices = table_futures.to_pandas()
    df_futures_prices.rename(columns={'time': 'Date'}, inplace=True)
    df_futures_prices['Date'] = pd.to_datetime(df_futures_prices['Date']).dt.tz_localize(None)
    df_futures_prices = df_futures_prices.sort_values('Date').reset_index(drop=True)
    print(f"[PPCI] Loaded {len(df_futures_prices)} futures price rows")
except Exception as _e:
    print(f"[PPCI] Could not load futures prices: {_e}")
    df_futures_prices = pd.DataFrame(columns=['Date'])

# Fetch FRED macro data (VIX, USD Index etc.) – stored by pipeline in macro_by_date
query_macro = """
SELECT *
FROM macro_by_date
WHERE time >= now() - INTERVAL '4 years'
"""
print("Fetching macro data (FRED) from InfluxDB v3...")
try:
    table_macro = client.query(query=query_macro, language="sql")
    df_macro = table_macro.to_pandas()
    df_macro.rename(columns={'time': 'Date'}, inplace=True)
    df_macro['Date'] = pd.to_datetime(df_macro['Date']).dt.tz_localize(None)
    df_macro = df_macro.sort_values('Date').reset_index(drop=True)
    print(f"[FRED] Loaded {len(df_macro)} macro rows")
except Exception as _e:
    print(f"[FRED] Could not load macro data: {_e}")
    df_macro = pd.DataFrame(columns=['Date'])

# Close the client
client.close()

# Rename columns for convenience
# v3 uses 'time' instead of '_time', data is already pivoted
df_pivoted.rename(columns={'time': 'Date', 'market_names': 'Market Names'}, inplace=True)

# Berechnung der Spalte 'Total Number of Traders'
df_pivoted = df_pivoted.sort_values(['Market Names', 'Date'])

# 1) Total Traders (TTF)
TOTAL_TRADERS_COL = 'Total Traders'
df_pivoted['Total Number of Traders'] = df_pivoted[TOTAL_TRADERS_COL]

# 2) Anteil Trader in Gruppe (nicht Open Interest!)
df_pivoted['MM_Long_share']  = df_pivoted['Traders M Money Long']  / df_pivoted['Total Number of Traders']
df_pivoted['MM_Short_share'] = df_pivoted['Traders M Money Short'] / df_pivoted['Total Number of Traders']

def clustering_0_100(s, window=52, minp=10):
    rmin = s.rolling(window, min_periods=minp).min()
    rmax = s.rolling(window, min_periods=minp).max()
    denom = (rmax - rmin).replace(0, np.nan)
    out = 100 * (s - rmin) / denom
    return out.clip(0, 100)

# 3) 1 Jahr = ~52 Wochen, und pro Markt (keine Markt-Mischung)
df_pivoted['Long Clustering'] = (
    df_pivoted.groupby('Market Names')['MM_Long_share']
    .transform(lambda s: clustering_0_100(s, window=52))
)

df_pivoted['Short Clustering'] = (
    df_pivoted.groupby('Market Names')['MM_Short_share']
    .transform(lambda s: clustering_0_100(s, window=52))
)

df_pivoted['Rolling Min'] = df_pivoted['Producer/Merchant/Processor/User Long'].rolling(365, min_periods=1).min()
df_pivoted['Rolling Max'] = df_pivoted['Producer/Merchant/Processor/User Long'].rolling(365, min_periods=1).max()

# Define size categories for traders
df_pivoted['Trader Size'] = pd.cut(
    df_pivoted['Total Number of Traders'],
    bins=[0, 50, 100, 150],
    labels=['≤ 50 Traders', '51–100 Traders', '101–150 Traders']
)

print(df_pivoted.columns)  # Zeigt alle Spalten in df_pivoted
print(df_pivoted.head())   # Zeigt die ersten Zeilen


# Additional calculations for the new graphs
df_pivoted['Total Long Traders'] = df_pivoted[['Traders Prod/Merc Short', 'Traders Swap Long', 'Traders M Money Long']].sum(axis=1)
df_pivoted['Total Short Traders'] = df_pivoted[['Traders Prod/Merc Short', 'Traders Swap Short', 'Traders M Money Short']].sum(axis=1)
df_pivoted['Long Position Size'] = df_pivoted['Producer/Merchant/Processor/User Long']
df_pivoted['Short Position Size'] = df_pivoted['Producer/Merchant/Processor/User Short']
df_pivoted['MML Position Size'] = (
    df_pivoted['Managed Money Long'] / df_pivoted['Traders M Money Long']
).replace([np.inf, -np.inf], np.nan)
df_pivoted['MMS Position Size'] = (
    df_pivoted['Managed Money Short'] / df_pivoted['Traders M Money Short']
).replace([np.inf, -np.inf], np.nan)

df_pivoted['Net Short Position Size'] = (
    df_pivoted['Short Position Size'] - df_pivoted['Long Position Size']
)
df_pivoted['PMPUL Position Size'] = (
    df_pivoted['Producer/Merchant/Processor/User Long'] / df_pivoted['Traders Prod/Merc Long']
).replace([np.inf, -np.inf], np.nan)

df_pivoted['PMPUS Position Size'] = (
    df_pivoted['Producer/Merchant/Processor/User Short'] / df_pivoted['Traders Prod/Merc Short']
).replace([np.inf, -np.inf], np.nan)
df_pivoted['SDL Position Size'] = (
    df_pivoted['Swap Dealer Long'] / df_pivoted['Traders Swap Long']
).replace([np.inf, -np.inf], np.nan)

df_pivoted['SDS Position Size'] = (
    df_pivoted['Swap Dealer Short'] / df_pivoted['Traders Swap Short']
).replace([np.inf, -np.inf], np.nan)
df_pivoted['ORL Position Size'] = (
    df_pivoted['Other Reportables Long'] / df_pivoted['Traders Other Rept Long']
).replace([np.inf, -np.inf], np.nan)

df_pivoted['ORS Position Size'] = (
    df_pivoted['Other Reportables Short'] / df_pivoted['Traders Other Rept Short']
).replace([np.inf, -np.inf], np.nan)

df_pivoted['MML Long OI'] = df_pivoted['Managed Money Long']
df_pivoted['MML Short OI'] = -df_pivoted['Managed Money Short']
df_pivoted['MMS Long OI'] = df_pivoted['Managed Money Long']
df_pivoted['MMS Short OI'] = -df_pivoted['Managed Money Short']
df_pivoted['MML Traders'] = df_pivoted['Traders M Money Long']
df_pivoted['MMS Traders'] = df_pivoted['Traders M Money Short']

max_bubble_size = 100
max_oi = max(df_pivoted['MML Long OI'].max(), abs(df_pivoted['MML Short OI'].max()))
max_oi = max(df_pivoted['MMS Short OI'].max(), abs(df_pivoted['MML Short OI'].max()))

sizeref = 2. * max_oi / (max_bubble_size**3.2)

# Calculate relative concentration for each trader group
df_pivoted['PMPUL Relative Concentration'] = df_pivoted['Producer/Merchant/Processor/User Long'] - df_pivoted['Producer/Merchant/Processor/User Short']
df_pivoted['PMPUS Relative Concentration'] = df_pivoted['Producer/Merchant/Processor/User Short'] - df_pivoted['Producer/Merchant/Processor/User Long']
df_pivoted['SDL Relative Concentration'] = df_pivoted['Swap Dealer Long'] - df_pivoted['Swap Dealer Short']
df_pivoted['SDS Relative Concentration'] = df_pivoted['Swap Dealer Short'] - df_pivoted['Swap Dealer Long']
df_pivoted['MML Relative Concentration'] = df_pivoted['Managed Money Long'] - df_pivoted['Managed Money Short']
df_pivoted['MMS Relative Concentration'] = df_pivoted['Managed Money Short'] - df_pivoted['Managed Money Long']
df_pivoted['ORL Relative Concentration'] = df_pivoted['Other Reportables Long'] - df_pivoted['Other Reportables Short']
df_pivoted['ORS Relative Concentration'] = df_pivoted['Other Reportables Short'] - df_pivoted['Other Reportables Long']

# Columns for the number of traders for each group
df_pivoted['PMPUL Traders'] = df_pivoted['Traders Prod/Merc Long']
df_pivoted['PMPUS Traders'] = df_pivoted['Traders Prod/Merc Short']
df_pivoted['SDL Traders'] = df_pivoted['Traders Swap Long']
df_pivoted['SDS Traders'] = df_pivoted['Traders Swap Short']
df_pivoted['MML Traders'] = df_pivoted['Traders M Money Long']
df_pivoted['MMS Traders'] = df_pivoted['Traders M Money Short']
df_pivoted['ORL Traders'] = df_pivoted['Traders Other Rept Long']
df_pivoted['ORS Traders'] = df_pivoted['Traders Other Rept Short']

# Determine the quarter for each date
df_pivoted['Quarter'] = df_pivoted['Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})

# Calculate a global sizeref to ensure consistency across markets
max_bubble_size = 100  # Adjusted for better visualization
max_oi = max(df_pivoted[['PMPUL Relative Concentration', 'PMPUS Relative Concentration', 
                         'SDL Relative Concentration', 'SDS Relative Concentration', 
                         'MML Relative Concentration', 'MMS Relative Concentration', 
                         'ORL Relative Concentration', 'ORS Relative Concentration']].max().max(),
             abs(df_pivoted[['PMPUL R'
                             'elative Concentration', 'PMPUS Relative Concentration',
                             'SDL Relative Concentration', 'SDS Relative Concentration', 
                             'MML Relative Concentration', 'MMS Relative Concentration', 
                             'ORL Relative Concentration', 'ORS Relative Concentration']].min().min()))
sizeref = 2. * max_oi / (max_bubble_size**2.5)

min_bubble_size = 10  # Set minimum bubble size

# Add Year column for color coding
df_pivoted['Year'] = df_pivoted['Date'].dt.year

# Calculate Net OI for Managed Money (MM)
df_pivoted['MM Net OI'] = df_pivoted['Managed Money Long'] - df_pivoted['Managed Money Short']

# Calculate Net Number of Traders for MM
df_pivoted['MM Net Traders'] = df_pivoted['Traders M Money Long'] - df_pivoted['Traders M Money Short']

# Define the default end date (most recent date)
default_end_date = df_pivoted['Date'].max()

# Define the default start date (6 months prior to the end date)
default_start_date = default_end_date - timedelta(days=182)


# ---------------------------------------------------------------------------
# PPCI – Positioning Price Concentration Indicator
# Preisspalten-Mapping: Substring im Market-Namen (UPPERCASE) →
# Feldname in der futures_prices-Tabelle (continuous contract als
# 2nd-Nearby-Proxy; yfinance GC=F etc. rollt automatisch).
# Annahme: kein direkter 2nd-Nearby-Ticker verfügbar – continuous contract
# liefert einen plausiblen Preis-Proxy für die Y-Achse.
# ---------------------------------------------------------------------------
_PPCI_MARKET_TO_COL = {
    'GOLD':      'gold_close',
    'SILVER':    'silver_close',
    'COPPER':    'copper_close',
    'PLATINUM':  'platinum_close',
    'PALLADIUM': 'palladium_close',
}


def _ppci_get_price_col(market_name: str):
    """Gibt den Spaltennamen in df_futures_prices zurück, oder None."""
    mn = (market_name or '').upper()
    for key, col in _PPCI_MARKET_TO_COL.items():
        if key in mn:
            return col
    return None


# Kontraktgrössen
_PPCI_CONTRACT_SIZES = {
    'GOLD':      100,    # troy ounces
    'SILVER':    5000,   # troy ounces
    'PLATINUM':  50,     # troy ounces
    'PALLADIUM': 100,    # troy ounces
    'COPPER':    25000,  # pounds
}


def _ppci_get_contract_size(market_name: str) -> float:
    """Gibt die Kontraktgröße (Einheiten/Kontrakt) zurück, oder 1 als Fallback."""
    mn = (market_name or '').upper()
    for key, size in _PPCI_CONTRACT_SIZES.items():
        if key in mn:
            return float(size)
    return 1.0


def get_global_xaxis():
    return dict(
        tickmode='array',
        tickvals=df_pivoted['Date'].dt.year.unique(),
        ticktext=[str(year) for year in df_pivoted['Date'].dt.year.unique()],
        showgrid=True,
        ticks="outside",
        tickangle=45
    )

global_xaxis = dict(
    tickmode='array',
    tickvals=df_pivoted['Date'].dt.year.unique(),  # Unique years
    ticktext=[str(year) for year in df_pivoted['Date'].dt.year.unique()],  # Format as strings
    showgrid=True,
    ticks="outside",
    tickangle=45  # Rotate for better visibility
)

def add_last_point_highlight(fig, df, x_col, y_col, inner_size=10, outer_line_width=4, outer_color='red', inner_color='black'):
    if not df.empty:  # Sicherstellen, dass die Daten nicht leer sind
        last_point = df.iloc[-1]

        # Innerer Punkt mit rotem Rand
        fig.add_trace(go.Scatter(
            x=[last_point[x_col]],
            y=[last_point[y_col]],
            mode='markers',
            marker=dict(
                size=inner_size,  # Größe des inneren Punkts
                color=inner_color,  # Farbe des inneren Punkts
                opacity=1.0,
                line=dict(
                    width=outer_line_width,  # Breite des äußeren Rands
                    color=outer_color  # Farbe des äußeren Rands
                )
            ),
            showlegend=False  # Spur nicht in der Legende anzeigen
        ))

def safe_sizes(series, exp=2.2, min_px=0):
    s = pd.to_numeric(series, errors='coerce').clip(lower=0)
    s = s.pow(1/exp).fillna(0)
    s = s * 0.7
    if min_px > 0:
        s = s + min_px
    return s

def dynamic_bubble_sizes(series, steps=5):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return [50, 100, 150]  # Fallback

    max_val = s.max()
    # Aufrunden auf "schöne" Zahl
    magnitude = 10 ** (len(str(int(max_val))) - 1)
    max_rounded = int(np.ceil(max_val / magnitude) * magnitude)

    # Gleichmäßig verteilte Werte
    return np.linspace(max_rounded / steps, max_rounded, steps, dtype=int).tolist()

def col(df, name):
    if name in df:
        return pd.to_numeric(df[name], errors='coerce')
    return pd.Series(np.nan, index=df.index, dtype='float64')

def safe_colors(series):
    return pd.to_numeric(series, errors='coerce').fillna(0)

# Function to calculate medians
def calculate_medians(df):
    median_oi = df['MM Net OI'].median()
    median_traders = df['MM Net Traders'].median()
    return median_oi, median_traders

# Function to calculate the scaling factors for long and short positions
def calculate_scaling_factors(df):
    max_long_position_size = df['Long Position Size'].max()
    max_short_position_size = df['Short Position Size'].max()
    long_scaling_factor = max_long_position_size / 50  # Adjust divisor as needed
    short_scaling_factor = max_short_position_size / 50  # Adjust divisor as needed
    return long_scaling_factor, short_scaling_factor

# Function to calculate concentration and clustering ranges
def calculate_ranges(agg_df, indicator):
    if indicator == 'MML':
        concentration_col = 'MML Relative Concentration'
        clustering_col = 'Long Clustering'
    elif indicator == 'MMS':
        concentration_col = 'MMS Relative Concentration'
        clustering_col = 'Short Clustering'
    else:
        raise ValueError("Invalid indicator. Must be 'MML' or 'MMS'.")

    # Filter to keep only numeric columns
    agg_df = agg_df.select_dtypes(include='number')

    # Calculate Concentration Range
    concentration_range = (agg_df[concentration_col] - agg_df[concentration_col].min()) / (agg_df[concentration_col].max() - agg_df[concentration_col].min())

    # Calculate Clustering Range
    clustering_range = (agg_df[clustering_col] - agg_df[clustering_col].min()) / (agg_df[clustering_col].max() - agg_df[clustering_col].min())

    return concentration_range * 100, clustering_range * 100

def nz(series):
    return pd.to_numeric(series, errors='coerce')

def rel_concentration(oi_long, oi_short, total_oi):
    oiL = nz(oi_long)
    oiS = nz(oi_short)
    tot = nz(total_oi).replace(0, np.nan)  # Division durch 0 vermeiden
    return 100.0 * ((oiL / tot) - (oiS / tot))

def scaled_diameters(vals, min_px=6, max_px=26):

    # In ein float-Array konvertieren (verträglich mit Series/ndarray/list/scalar)
    v = np.asarray(vals, dtype=float)

    # Nicht-finite durch 0 ersetzen
    v = np.where(np.isfinite(v), v, 0.0)

    if v.size == 0:
        return np.array([], dtype=float)

    lo = np.nanmin(v)
    hi = np.nanmax(v)

    # Falls alle Werte gleich (oder leer), mittlere Größe verwenden
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(v, (min_px + max_px) / 2.0, dtype=float)

    # Linear skalieren auf Durchmesser (Pixel)
    return np.interp(v, (lo, hi), (min_px, max_px))

def scaled_diameters_rank(vals, min_px=6, max_px=45, gamma=0.8):

    s = pd.to_numeric(pd.Series(vals), errors='coerce').fillna(0).clip(lower=0)

    # alles gleich / keine Variation -> konstante Größe
    if s.nunique(dropna=False) <= 1:
        return np.full(len(s), (min_px + max_px) / 2.0, dtype=float)

    # Rang/Perzentil (0..1)
    p = s.rank(pct=True, method='average').to_numpy(dtype=float)

    # in Pixel mappen
    return (min_px + (p ** gamma) * (max_px - min_px)).astype(float)

# Example calculation
median_oi, median_traders = calculate_medians(df_pivoted)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    external_scripts=["https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"]
)

# Layout of the app
app.layout = html.Div([
    dbc.NavbarSimple(
        brand="COT-Data Overview/Analysis Dashboard",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Market Overview"),
                dcc.Dropdown(
                    id='market-dropdown',
                    options=[{'label': market, 'value': market} for market in df_pivoted['Market Names'].unique()],
                    value='Palladium',  # Default value
                    style={'width': '100%'}
                ),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    start_date=df_pivoted['Date'].min(),
                    end_date=df_pivoted['Date'].max(),
                    display_format='YYYY-MM-DD',
                    className='mb-4'
                ),
                dash_table.DataTable(
                    id='overview-table',
                    columns=[
                        {'name': 'Trader Group', 'id': 'Trader Group'},
                        {'name': 'Positions (OI)', 'id': 'Positions', 'presentation': 'markdown'},  # <-- neu
                        {'name': 'Δ Long %', 'id': 'Difference (Long %)'},
                        {'name': 'Δ Short %', 'id': 'Difference (Short %)'},
                        {'name': 'Δ Spread %', 'id': 'Difference (Spread %)'},
                        {'name': 'Total Traders', 'id': 'Total Traders'},
                        {'name': '% of Traders', 'id': '% of Traders'},
                        {'name': 'Number of Traders', 'id': 'Number of Traders', 'presentation': 'markdown'},
                    ],
                    markdown_options={"html": True},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold',
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    css=[
                        {"selector": 'th[data-dash-column="Number of Traders"]',
                         "rule": "white-space: normal;"},
                        {"selector": 'th[data-dash-column="Number of Traders"] .column-header-name',
                         "rule": "display: block;"},
                        {"selector": 'th[data-dash-column="Number of Traders"]::after',
                         "rule": (
                             "content: 'Long   Short   Spread';"
                             "display: block;"
                             "margin-top: 4px;"
                             "font-size: 11px;"
                             "color: #444;"
                             "line-height: 1.3;"
                             "padding-left: 18px;"
                             "word-spacing: 26px;"
                             "background-image: "
                             "radial-gradient(circle, #2ca02c 0, #2ca02c 100%),"
                             "radial-gradient(circle, #d62728 0, #d62728 100%),"
                             "radial-gradient(circle, #1f77b4 0, #1f77b4 100%);"
                             "background-repeat: no-repeat;"
                             "background-size: 10px 10px, 10px 10px, 10px 10px;"
                             "background-position: 2px 55%, 60px 55%, 120px 55%;"
                         )},
                        {"selector": 'th[data-dash-column="Positions"]',
                         "rule": "white-space: normal;"},
                        {"selector": 'th[data-dash-column="Positions"] .column-header-name',
                         "rule": "display: block;"},
                        {"selector": 'th[data-dash-column="Positions"]::after',
                         "rule": (
                             "content: 'Long   Short   Spread';"
                             "display: block;"
                             "margin-top: 4px;"
                             "font-size: 11px;"
                             "color: #444;"
                             "line-height: 1.3;"
                             "padding-left: 18px;"
                             "word-spacing: 26px;"
                             "background-image: "
                             "radial-gradient(circle, #2ca02c 0, #2ca02c 100%),"
                             "radial-gradient(circle, #d62728 0, #d62728 100%),"
                             "radial-gradient(circle, #1f77b4 0, #1f77b4 100%);"
                             "background-repeat: no-repeat;"
                             "background-size: 10px 10px, 10px 10px, 10px 10px;"
                             "background-position: 2px 55%, 60px 55%, 120px 55%;"
                         )},
                    ],
                style_cell_conditional=[
                        {'if': {'column_id': 'Positions'},
                         'whiteSpace': 'normal', 'height': 'auto',
                         'minWidth': '260px', 'width': '260px', 'maxWidth': '260px'},

                        {'if': {'column_id': 'Number of Traders'},
                         'whiteSpace': 'normal', 'height': 'auto',
                         'minWidth': '260px', 'width': '260px', 'maxWidth': '260px'}
                    ],
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{Difference (Long %)} < 0',
                                'column_id': 'Difference (Long %)'
                            },
                            'color': 'red'
                        },
                        {
                            'if': {
                                'filter_query': '{Difference (Long %)} > 0',
                                'column_id': 'Difference (Long %)'
                            },
                            'color': 'green'
                        },
                        {
                            'if': {
                                'filter_query': '{Difference (Short %)} < 0',
                                'column_id': 'Difference (Short %)'
                            },
                            'color': 'red'
                        },
                        {
                            'if': {
                                'filter_query': '{Difference (Short %)} > 0',
                                'column_id': 'Difference (Short %)'
                            },
                            'color': 'green'
                        },
                        {
                            'if': {
                                'filter_query': '{Difference (Spread %)} < 0',
                                'column_id': 'Difference (Spread %)'
                            },
                            'color': 'red'
                        },
                        {
                            'if': {
                                'filter_query': '{Difference (Spread %)} > 0',
                                'column_id': 'Difference (Spread %)'
                            },
                            'color': 'green'
                        },
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '5px',
                        'border': '1px solid grey',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                )
            ], width=12)
        ]),

       html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("Clustering Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **Clustering Indicator** misst, wie viele Trader eine bestimmte Long- oder Short-Position halten, 
                        ausgedrückt als Prozentsatz aller Trader im Markt. Er ist damit ein Indikator für Marktstimmung und „Herdentrieb“.

                        Das **Ziel des Indikators** ist es, das Mass an „Crowding“ in einem Markt sichtbar zu machen – also wie viele 
                        Trader sich in dieselbe Richtung positionieren. Er ist unabhängig von der Positionsgrösse und passt sich dadurch gut an
                        regulatorische Beschränkungen wie Positionslimits oder Diversifikationsauflagen an.

                        **Farbskala:** Die Farbe der Punkte zeigt den *Clustering-Wert in %*. Dieser Wert zeigt, wie
                        stark sich Trader im Verhältnis zur historischen Bandbreite (ein Jahr) in einer Long- oder Short-Position 
                        konzentrieren. Ein hoher Wert bedeutet also, dass sich besonders viele Trader in derselben Richtung
                        positionieren.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(dcc.Markdown(r"""
                            **Berechnung Long-Clustering (Money Manager):**

                            $$
                            \mathrm{Clustering}^{\mathrm{(Long)}}_{\mathrm{MM}}(\%)=
                            \frac{\mathrm{Number\ of\ traders}^{\mathrm{(Long)}}_{\mathrm{MM}}}
                            {\mathrm{Total\ number\ of\ traders}}
                            $$
                            """, mathjax=True), width=12, lg=6),

                            dbc.Col(dcc.Markdown(r"""
                            **Berechnung Short-Clustering (Money Manager):**

                            $$
                            \mathrm{Clustering}^{\mathrm{(Short)}}_{\mathrm{MM}}(\%)=
                            \frac{\mathrm{Number\ of\ traders}^{\mathrm{(Short)}}_{\mathrm{MM}}}
                            {\mathrm{Total\ number\ of\ traders}}
                            $$
                            """, mathjax=True), width=12, lg=6),
                        ], className="mb-2"),

                        dcc.Markdown(r"""
                        **Bedeutung der Abkürzungen / Begriffe:**
                        - **MM:** Money Manager
                        - **Number of traders $\mathrm{MM}_{\mathrm{Long}}$:** Anzahl MM-Trader mit Long-Positionen
                        - **Number of traders $\mathrm{MM}_{\mathrm{Short}}$:** Anzahl MM-Trader mit Short-Positionen
                        - **Total number of traders:** Gesamtanzahl Trader im Markt
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.Graph(id='long-clustering-graph'),
        html.Div([], style={'marginTop': '10px'}),
        dcc.Graph(id='short-clustering-graph'),
        html.Br(),
    ], width=12)
]),

        html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("Position Size Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(
                            r"""
                            Der **Position Size Indicator** misst die durchschnittliche Grösse der Positionen einzelner Trader, 
                            indem die gesamte Positionsgrösse durch die Anzahl der beteiligten Trader geteilt wird. Dadurch wird sichtbar, 
                            wie stark die Überzeugung (*conviction*) innerhalb einer Tradergruppe ist.

                            Das **Ziel des Indikators** ist es, die durchschnittliche Positionsgrösse und damit die Intensität des Engagements von Tradern transparenter zu machen. 
                            Er kombiniert Daten zu *Open Interest* und *Traderanzahl*, um Rückschlüsse auf die Verteilung von Positionen entlang der Fälligkeiten 
                            (*down the curve*) zu ziehen. Zudem lassen sich über Positionslimits erkennen, wie stark Positionen konzentriert sind und welche 
                            Auswirkungen ein Abbau dieser Positionen auf Preise und Marktstruktur haben könnte.

                            **Farbskala:** Die Punktfarbe zeigt die *durchschnittliche Positionsgrösse* in der jeweiligen Gruppe. 
                            Helle Farben = grössere Positionen pro Trader, dunkle Farben = kleinere Positionen.
                            """,
                            mathjax=True
                        ),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(
                            r"""
                            **Berechnung:**

                            $$
                            \text{Position Size}_{G} =
                            \frac{\text{Open Interest}_{G}}
                            {\text{Number of Traders}_{G}}
                            $$

                            wobei
                            $$
                            G \in \{\mathrm{MM}\text{-}L,\, \mathrm{MM}\text{-}S,\, \mathrm{PMPU}\text{-}L,\, \mathrm{PMPU}\text{-}S,\, \mathrm{SD}\text{-}L,\, \mathrm{SD}\text{-}S,\, \mathrm{OR}\text{-}L,\, \mathrm{OR}\text{-}S\}
                            $$

                            **Bedeutung der Abkürzungen:**
                            - **PMPU:** Producer/Merchant/Processor/User
                            - **SD:** Swap Dealer
                            - **MM:** Managed Money
                            - **OR:** Other Reportables
                            - **L:** Long Positionen
                            - **S:** Short Positionen
                            """,
                            mathjax=True
                        ),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),
    ], width=12)
]),

dbc.Row([dbc.Col([html.H2("Producer/Merchant/Processor/User (PMPU)")], width=12)]),
dbc.Row([
    dbc.Col([dcc.Graph(id='pmpu-long-position-size-graph')], width=12),
    dbc.Col([dcc.Graph(id='pmpu-short-position-size-graph')], width=12),
]),
dbc.Row([dbc.Col([html.H2("Swap Dealers")], width=12)]),
dbc.Row([
    dbc.Col([dcc.Graph(id='sd-long-position-size-graph')], width=12),
    dbc.Col([dcc.Graph(id='sd-short-position-size-graph')], width=12),
]),
dbc.Row([dbc.Col([html.H2("Money Managers")], width=12)]),
dbc.Row([
    dbc.Col([dcc.Graph(id='long-position-size-graph')], width=12),
    dbc.Col([dcc.Graph(id='short-position-size-graph')], width=12),
]),
dbc.Row([dbc.Col([html.H2("Other Reportables")], width=12)]),
dbc.Row([
    dbc.Col([dcc.Graph(id='or-long-position-size-graph')], width=12),
    dbc.Col([dcc.Graph(id='or-short-position-size-graph')], width=12),
]),

        html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("Dry Powder Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Dry Powder (DP)** ist eine Methode zur Visualisierung der Positionierung in Rohstoffmärkten. 
                        Dabei wird die Grösse der Long- und Short-Positionen (*Open Interest*) mit der Anzahl der Trader 
                        in einer bestimmten Gruppe (z. B. Money Managers) in Beziehung gesetzt.

                        Das **Ziel der DP-Analyse** ist es, einschätzen zu können, ob bestehende Positionen noch ausgebaut werden 
                        können oder ob sie anfällig für Liquidationen sind. DP-Indikatoren werden in Diagrammen dargestellt 
                        und können direkt als Handelssignale genutzt werden, um Marktchancen und Risiken besser zu bewerten.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Berechnung:**

                        Achsen (Zeitpunkt $t$):
                        - **x-Achse:** Anzahl der Trader in der jeweiligen Gruppe
                        - **y-Achse:** Grösse der offenen Positionen (Open Interest)

                        $$
                        x_{\mathrm{MML}}(t) = N_{\mathrm{MML}}(t), 
                        \qquad
                        x_{\mathrm{MMS}}(t) = N_{\mathrm{MMS}}(t)
                        $$

                        $$
                        y_{\mathrm{MML}}(t) = OI_{\mathrm{MML}}^{L}(t),
                        \qquad
                        y_{\mathrm{MMS}}(t) = OI_{\mathrm{MMS}}^{S}(t)\;(\text{im Plot negativ})
                        $$

                        **Bubble-Grösse:**  
                        Die Fläche der Bubbles zeigt, wie gross die Gesamtposition (Long + Short) im Verhältnis ist – je grösser die Bubble, desto mehr offene Kontrakte (Open Interest) liegen vor.

                        **Begriffe:**  
                        - $OI$ (*Open Interest*): Anzahl offener Kontrakte (Long bzw. Short) einer Gruppe zu einem Zeitpunkt
                        - $N_{\mathrm{MML}}, N_{\mathrm{MMS}}$: Anzahl Trader (Money Manager Long bzw. Short)
                        - **Farbkodierung:** Dunkelblau = MML-Wolke (Long-Seite), Hellblau = MMS-Wolke (Short-Seite)
                        - **Schwarzer Punkt:** jeweils die **aktuellste Woche**
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.Graph(id='dry-powder-indicator-graph')
    ], width=12)
]),

html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("DP Notional Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **Dry Powder Notional Indicator** misst die aggregierte Long- bzw. Short-Exponierung 
                        einer bestimmten Tradergruppe in Notional-Dollar, indem das Open Interest der Gruppe mit der 
                        Kontraktgrösse und dem zugrunde liegenden Futures-Preis multipliziert wird. Dadurch wird sichtbar, 
                        wie gross das gesamte finanzielle Exposure einer Gruppe im Markt ist und wie stark sie kapitalmässig engagiert ist.

                        Das **Ziel des Indikators** ist es, die absolute Markt-Exponierung einer Tradergruppe in USD sichtbar 
                        zu machen. Er hilft zu beurteilen, ob Veränderungen im Engagement einer Gruppe primär auf eine veränderte 
                        Anzahl Trader oder auf ein höheres bzw. tieferes aggregiertes Positionsvolumen zurückzuführen sind. 
                        Dadurch lassen sich Rückschlüsse auf die finanzielle Bedeutung und das Marktgewicht einzelner Tradergruppen ziehen.

                        **Farbskala:** Die Punktfarbe unterscheidet die jeweilige Tradergruppe. In der dargestellten Ausprägung 
                        steht dunkelblau für Managed Money Long (MML) und hellblau für Managed Money Short (MMS).
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Berechnung:**

                        $$
                        x_G(t) = \mathrm{Traders}_G(t)
                        $$

                        $$
                        y_G(t) = \mathrm{DP\ Notional}_G(t)
                        = \mathrm{Position}_G(t)\times \mathrm{ContractSize}\times \mathrm{Price}(t)
                        $$

                        **Bedeutung der Abkürzungen / Begriffe:**
                        - **$G$:** betrachtete Tradergruppe, $\mathrm{MML}$ oder $\mathrm{MMS}$
                        - **$\mathrm{Traders}_G(t)$:** Anzahl Trader der Gruppe $G$ zum Zeitpunkt $t$
                        - **$\mathrm{Position}_G(t)$:** aggregierte Position der Gruppe $G$ zum Zeitpunkt $t$ (in Kontrakten)
                        - **$\mathrm{ContractSize}$:** Kontraktgrösse des jeweiligen Futures
                        - **$\mathrm{Price}(t)$:** Futures-Preis zum Zeitpunkt $t$
                        - **$\mathrm{DP\ Notional}_G(t)$:** aggregiertes Notional-Exposure der Gruppe $G$ in USD
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.Graph(id='dp-notional-indicator-graph'),
        html.Br(),
    ], width=12)
]),

html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("DP Time Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **Dry Powder Time Indicator** zeigt, wie sich die Long- bzw. Short-Konzentration 
                        einer bestimmten Tradergruppe in Abhängigkeit von der Anzahl Trader über die Zeit entwickelt. 
                        Dadurch wird sichtbar, wie stark eine Gruppe relativ zum gesamten Markt positioniert ist und 
                        wie sich diese Positionierung im historischen Verlauf verändert.

                        Das **Ziel des Indikators** ist es, die zeitliche Entwicklung der Marktpositionierung einer 
                        Tradergruppe sichtbar zu machen. Er hilft zu beurteilen, ob sich Konzentrationsmuster in Phasen 
                        mit vielen oder wenigen Tradern wiederholen und wie sich die aktuelle Positionierung im Vergleich 
                        zu früheren Jahren einordnen lässt. Dadurch lassen sich historische Muster, Verschiebungen in der 
                        Marktstruktur und mögliche Extremphasen erkennen.

                        **Farbskala:** Die Punktfarbe codiert das jeweilige Jahr der Beobachtung. Dadurch wird sichtbar, 
                        aus welcher Zeitperiode ein Punkt stammt und wie sich die Positionierung über die Jahre entwickelt.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(dcc.Markdown(r"""
                            **Für Managed Money Long (MML):**

                            $$
                            \mathrm{DP\ Time}_{\mathrm{MML}}=
                            \frac{\mathrm{Open\ Interest}_{\mathrm{MML}}}
                            {\mathrm{Total\ Open\ Interest}} \cdot 100
                            $$
                            """, mathjax=True), width=12, lg=6),

                            dbc.Col(dcc.Markdown(r"""
                            **Für Managed Money Short (MMS):**

                            $$
                            \mathrm{DP\ Time}_{\mathrm{MMS}}=
                            -\,\frac{\mathrm{Open\ Interest}_{\mathrm{MMS}}}
                            {\mathrm{Total\ Open\ Interest}} \cdot 100
                            $$
                            """, mathjax=True), width=12, lg=6),
                        ], className="mb-2"),

                        dcc.Markdown(r"""
                        **Bedeutung der Abkürzungen / Begriffe:**
                        - **MML:** Managed Money Long
                        - **MMS:** Managed Money Short
                        - **$\mathrm{Open\ Interest}_{\mathrm{MML}}$:** Open Interest der Managed-Money-Long-Positionen
                        - **$\mathrm{Open\ Interest}_{\mathrm{MMS}}$:** Open Interest der Managed-Money-Short-Positionen
                        - **Total Open Interest:** gesamtes Open Interest des betrachteten Futures-Marktes
                        - **Negatives Vorzeichen bei MMS:** dient der separaten Darstellung der Short-Seite im Plot
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.Graph(id='dp-time-indicator-graph'),
        html.Br(),
    ], width=12)
]),

html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("DP Price Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **Dry Powder Price Indicator** zeigt, wie sich das Long- bzw. Short-Open Interest 
                        einer bestimmten Tradergruppe in Abhängigkeit von der Anzahl Trader und dem zugrunde 
                        liegenden Preisniveau verteilt. Dadurch wird sichtbar, bei welchen Preisniveaus eine 
                        Gruppe besonders stark oder schwach positioniert ist und wie sich diese Positionierung 
                        im Verhältnis zur Zahl der beteiligten Trader verändert.

                        Das **Ziel des Indikators** ist es, die Positionierung einer Tradergruppe im Zusammenhang 
                        mit dem Marktpreis sichtbar zu machen. Er hilft zu beurteilen, ob hohe oder tiefe Long- 
                        bzw. Short-Positionierungen eher bei bestimmten Preisniveaus auftreten und ob diese von 
                        vielen oder wenigen Tradern getragen werden. Dadurch lassen sich typische Preisbereiche 
                        identifizieren, in denen eine Gruppe besonders aktiv ist.

                        **Farbskala:** Die Punktfarbe zeigt das jeweilige Preisniveau des Continuous Front-Monts Features 
                        (2nd nearby). Helle Farben stehen für tiefere Preise, dunklere Farben für höhere Preise.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(dcc.Markdown(r"""
                            **Für die Long-Seite:**

                            $$
                            \mathrm{DP\ Price}_{G}(t)=\mathrm{Open\ Interest}_{G}(t)
                            $$
                            """, mathjax=True), width=12, lg=6),

                            dbc.Col(dcc.Markdown(r"""
                            **Für die Short-Seite:**

                            $$
                            \mathrm{DP\ Price}_{G}(t)=\mathrm{Open\ Interest}_{G}(t)
                            $$
                            """, mathjax=True), width=12, lg=6),
                        ], className="mb-2"),

                        dcc.Markdown(r"""
                        **Bedeutung der Abkürzungen / Begriffe:**
                        - **$G$:** betrachtete Tradergruppe auf der gewählten Marktseite, $\mathrm{PMPUL}$ oder $\mathrm{PMPUS}$
                        - **$\mathrm{Open\ Interest}_{G}(t)$:** Open Interest der betrachteten Gruppe $G$ zum Zeitpunkt $t$
                        - **x-Achse:** Anzahl Trader der betrachteten Gruppe
                        - **y-Achse:** Open Interest der betrachteten Gruppe
                        - **Punktfarbe:** Preisniveau des Continuous Front-Monts Features zum Zeitpunkt $t$
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.RadioItems(
            id='dp-price-radio',
            options=[
                {'label': 'PMPU Long',  'value': 'PMPUL'},
                {'label': 'PMPU Short', 'value': 'PMPUS'},
            ],
            value='PMPUL',
            className='mb-4'
        ),
        dcc.Graph(id='dp-price-indicator-graph'),
        html.Br(),
    ], width=12)
]),

html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("DP Factor (VIX)"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **Dry Powder Factor (VIX) Indicator** zeigt, wie sich das Long- bzw. 
                        Short-Open-Interest einer bestimmten Tradergruppe in Abhängigkeit von der Anzahl 
                        Trader und einem externen Risikofaktor verteilt. Als externer Faktor wird der 
                        **Volatility Index (VIX)** verwendet, der die vom Markt erwartete Schwankungsintensität 
                        des S&P 500 für die nächsten 30 Tage abbildet. Dadurch wird sichtbar, ob eine Tradergruppe 
                        ihre Positionen eher in Phasen tiefer oder hoher erwarteter Marktvolatilität aufbaut.

                        Das **Ziel des Indikators** ist es, die Positionierung einer Tradergruppe im Zusammenhang 
                        mit dem allgemeinen Marktunsicherheitsniveau sichtbar zu machen. Er hilft zu beurteilen, 
                        ob hohe oder tiefe Long- bzw. Short-Positionierungen eher in Phasen erhöhter oder reduzierter 
                        Risikoaversion auftreten und ob diese von vielen oder wenigen Tradern getragen werden. Dadurch 
                        lassen sich mögliche Zusammenhänge zwischen externer Marktunsicherheit und der Positionierung 
                        im jeweiligen Rohstoffmarkt erkennen.

                        **Farbskala:** Die Punktfarbe zeigt das jeweilige Niveau des VIX. Helle Farben stehen für 
                        eine tiefere erwartete Volatilität, dunkelrote Farben für eine höhere erwartete Volatilität.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dbc.Row([
                            dbc.Col(dcc.Markdown(r"""
                            **Für Managed Money Long (MML):**

                            $$
                            \mathrm{DP\ Factor\ (VIX)}_{\mathrm{MML}}=
                            \mathrm{Open\ Interest}_{\mathrm{MML}}
                            $$
                            """, mathjax=True), width=12, lg=6),

                            dbc.Col(dcc.Markdown(r"""
                            **Für Managed Money Short (MMS):**

                            $$
                            \mathrm{DP\ Factor\ (VIX)}_{\mathrm{MMS}}=
                            \mathrm{Open\ Interest}_{\mathrm{MMS}}
                            $$
                            """, mathjax=True), width=12, lg=6),
                        ], className="mb-2"),

                        dcc.Markdown(r"""
                        **Bedeutung der Abkürzungen / Begriffe:**
                        - **MML:** Managed Money Long
                        - **MMS:** Managed Money Short
                        - **$\mathrm{Open\ Interest}_{\mathrm{MML}}$:** Open Interest der Managed-Money-Long-Positionen
                        - **$\mathrm{Open\ Interest}_{\mathrm{MMS}}$:** Open Interest der Managed-Money-Short-Positionen
                        - **x-Achse:** Anzahl Trader der betrachteten Gruppe
                        - **y-Achse:** Open Interest der betrachteten Gruppe
                        - **Punktfarbe:** Niveau des VIX zum Zeitpunkt $t$
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.RadioItems(
            id='dp-vix-radio',
            options=[
                {'label': 'MM Long',  'value': 'MML'},
                {'label': 'MM Short', 'value': 'MMS'},
            ],
            value='MML',
            className='mb-4'
        ),
        dcc.Graph(id='dp-vix-indicator-graph'),
        html.Br(),
    ], width=12)
]),

        html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("DP Relative Concentration Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(
                            r"""
                            Der **DP Relative Concentration Indicator** normalisiert Positionen 
                            anhand des Open Interest und stellt die Konzentration der Gruppen dar. Dadurch lassen sich verschiedene Märkte 
                            oder Gruppen innerhalb eines Marktes direkt vergleichen.

                            Das **Ziel des Indikators** ist es, die Positionierungsprofile von Märkten vollständig zu visualisieren und Unterschiede 
                            sichtbar zu machen – etwa zwischen verwandten Rohstoffen wie Mais und Sojabohnen oder zwischen WTI und Brent. 
                            Dadurch können Rückschlüsse auf zukünftige Marktbewegungen, Hedging-Verhalten und potenzielle Spreadausweitungen 
                            gezogen werden.
                            """,
                            mathjax=True
                        ),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(
                            r"""
                            **Berechnung:**

                            Achsen (Zeitpunkt $(t)$):
                            - **x-Achse:** Anzahl Trader in der jeweiligen Gruppe (Long oder Short)
                            - **y-Achse:** Relative Concentration $(RC_G(t))$, d. h. die Nettopositionierung der Gruppe $(G)$ relativ zum gesamten Open Interest

                            $$
                            x_G(t) = N_G(t),
                            \qquad
                            y_G(t) = RC_G(t)
                            $$

                            mit

                            $$
                            RC_G(t) = 100 \cdot \sigma_G \left( \frac{L_G(t)}{OI(t)} - \frac{S_G(t)}{OI(t)} \right)
                            $$

                            wobei
                            $$
                            G \in \{\mathrm{MM}\text{-}L,\, \mathrm{MM}\text{-}S,\, \mathrm{PMPU}\text{-}L,\, \mathrm{PMPU}\text{-}S,\, \mathrm{SD}\text{-}L,\, \mathrm{SD}\text{-}S,\, \mathrm{OR}\text{-}L,\, \mathrm{OR}\text{-}S\}
                            $$

                            und  
                            - $L_G(t)$: Long Open Interest der Gruppe $G$  
                            - $S_G(t)$: Short Open Interest der Gruppe $G$  
                            - $OI(t)$: Gesamtes Open Interest zum Zeitpunkt $t$  
                            - $N_G(t)$: Anzahl Trader (Long oder Short) der Gruppe $G$  
                            - $\sigma_G = +1$ für Long-Serien (MM-L, OR-L, PMPU-L, SD-L)  
                            - $\sigma_G = -1$ für Short-Serien (MM-S, OR-S, PMPU-S, SD-S)

                            **Begriffe:**  
                            - $OI$ (*Open Interest*): Anzahl aller offenen Kontrakte
                            - $N_G$: Anzahl Trader in Gruppe $G$
                            - $RC_G(t)$: Relative Concentration (in Prozentpunkten) einer Gruppe
                            - **Schwarzer Punkt:** markiert den Wert der **aktuellsten Woche** je Gruppe
                            """,
                            mathjax=True
                        ),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.Graph(id='dp-relative-concentration-graph')
    ], width=12)
]),

        html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("DP Seasonal Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **DP Seasonal Indicator** ist ein spezieller DP-Indikator, der saisonale Muster im Traderverhalten 
                        sichtbar macht. Dabei werden Positionen nicht nur nach Grösse und Anzahl der Trader, sondern zusätzlich 
                        nach Zeitabschnitten (z. B. Monate oder Quartale) dargestellt.

                        **Das Ziel dieses Indikators** ist es, saisonale Hedging-Muster oder Abweichungen davon zu erkennen. 
                        So lassen sich etwa typische Verhaltensweisen von Produzenten oder Konsumenten in bestimmten Jahreszeiten 
                        aufzeigen (z. B. stärkere Hedging-Aktivität im Winter bei Heizöl). Gleichzeitig hilft er, potenzielle 
                        Anomalien oder Unterabsicherungen zu identifizieren, die ein Risiko für Preisbewegungen darstellen könnten.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Berechnung:**

                        $$
                        x_q(t) = N_q(t), \qquad y_q(t) = RC_q(t)
                        $$

                        wobei

                        - $N_q(t)$: Anzahl der Trader im Quartal $q$ zum Zeitpunkt $t$
                        - $RC_q(t)$: *Relative Concentration* der Tradergruppe im Quartal $q$
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.Graph(id='dp-seasonal-indicator-graph')
    ], width=12)
]),

        html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("DP Net Indicator with Median"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **DP Net Indicator** kombiniert Informationen zu Netto-Open-Interest und Netto-Anzahl von Tradern. 
                        Dadurch lassen sich Abweichungen zwischen Positionsgrösse und Traderanzahl sichtbar machen, die Hinweise 
                        auf mögliche Wendepunkte im Markt geben können.

                        Das **Ziel dieses Indikators** ist es, ein klareres Bild der Netto-Positionierung zu liefern und Extremwerte 
                        besser einzuordnen. So können Situationen erkannt werden, in denen z. B. das Open Interest eine Long-Position 
                        zeigt, die Mehrheit der Trader aber Short positioniert ist. Zudem lassen sich auch Spread-Positionen analysieren, 
                        um einzuschätzen, ob diese sich in extremeren Marktphasen (z. B. Contango oder Backwardation) verstärken könnten.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Berechnung:**

                        Achsen (Zeitpunkt $(t)$):
                        - **x-Achse:** Netto-Anzahl Money-Manager-Trader
                        - **y-Achse:** Netto-Open-Interest der Money Manager
                        
                        $$
                        x(t)=N^{\text{Net}}(t)=N^{\text{Long}}(t)-N^{\text{Short}}(t),
                        \qquad
                        y(t)=OI^{\text{Net}}(t)=OI^{\text{Long}}(t)-OI^{\text{Short}}(t)
                        $$
                        
                        **Medians (gestrichelte Referenzlinien):**
                        $$
                        \widetilde{N}^{\text{Net}}=\operatorname{Median}_t\!\big(N^{\text{Net}}(t)\big),
                        \qquad
                        \widetilde{OI}^{\text{Net}}=\operatorname{Median}_t\!\big(OI^{\text{Net}}(t)\big)
                        $$
                        
                        **Variablen (mit Datenbezug):**
                        - $t$: Kalenderwoche/Beobachtungszeitpunkt innerhalb des gewählten Datumsbereichs
                        - $N^{\text{Long}}(t)$: Anzahl **Long-Trader (MM)** zum Zeitpunkt $t$
                        - $N^{\text{Short}}(t)$: Anzahl **Short-Trader (MM)** zum Zeitpunkt $t$
                        - $N^{\text{Net}}(t)$: **Netto-Traderzahl** $=\;N^{\text{Long}}(t)-N^{\text{Short}}(t)$
                        - $OI^{\text{Long}}(t)$: **Long-Open-Interest (MM)** zum Zeitpunkt $t$
                        - $OI^{\text{Short}}(t)$: **Short-Open-Interest (MM)** zum Zeitpunkt $t$
                        - $OI^{\text{Net}}(t)$: **Netto-Open-Interest** $=\;OI^{\text{Long}}(t)-OI^{\text{Short}}(t)$
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.Graph(id='dp-net-indicators-graph')
    ], width=12)
]),

        html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("DP Position Size Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **DP Position Size Indicator** verknüpft die durchschnittliche Positionsgrösse von Tradern 
                        mit der Preisentwicklung eines Rohstoffs. Dabei wird die Positionsgrösse (y-Achse) gegen die Anzahl der Trader 
                        (x-Achse) dargestellt, wobei die Farben die jeweilige Preisrange markieren.

                        Das **Ziel dieses Indikators** ist es, Zusammenhänge zwischen Positionsgrössen und Marktpreisen sichtbar zu machen. 
                        So lassen sich Muster erkennen, etwa dass Long-Trader bei tieferen Preisen grössere Positionen halten 
                        (stärkeres Engagement), während bei höheren Preisen die Traderzahl sinkt. Auf der Short-Seite hingegen treten 
                        oft uneinheitlichere Muster auf, was auf unterschiedliche Handelsstrategien wie Spread- oder 
                        Relative-Value-Trading hinweist.

                        Insgesamt hilft der Indikator, Unterschiede im Verhalten von Long- und Short-Tradern zu analysieren 
                        und Rückschlüsse auf ihre Handelsmotive (z. B. direktional vs. relative Value) zu ziehen.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Berechnung:**

                        Achsen:
                        $$
                        x_g(t)=N_g(t), \qquad y_g(t)=\mathrm{PS}_g(t)
                        $$
                        - $N_g(t)$: Anzahl der Trader einer Gruppe $g$ zum Zeitpunkt $t$
                        - $\mathrm{PS}_g(t)$: durchschnittliche Positionsgrösse je Trader einer Gruppe $g$ zum Zeitpunkt $t$

                        Farbcodierung:
                        $$
                        \text{color}_g(t)\;\propto\;\mathrm{OI}_g(t)
                        $$
                        - $\mathrm{OI}_g(t)$: Open Interest zum Zeitpunkt $t$, d. h. die gesamte Anzahl offener Kontrakte
                        - Die **Farbe eines Punktes** zeigt somit an, wie hoch das Open Interest in der jeweiligen Woche war
                          (je heller/gelber, desto höher das Open Interest)
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.RadioItems(
            id='mm-radio',
            options=[
                {'label': 'MML', 'value': 'MML'},
                {'label': 'MMS', 'value': 'MMS'}
            ],
            value='MML',
            className='mb-4'
        ),
        dcc.Graph(id='dp-position-size-indicator')
    ], width=12)
]),

        html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("DP Hedging Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **DP Hedging Indicators** erweitern die klassische DP-Analyse, indem sie mehrere Tradergruppen 
                        gleichzeitig betrachten – typischerweise Money Manager (MM) und Produzenten/Verbraucher (PMPU). 
                        So wird sichtbar, wie viel „Dry Powder“ (Spielraum für zusätzliche Positionen) eine Gruppe im Verhältnis 
                        zu einer anderen noch hat.

                        Das **Ziel dieser Indikatoren** ist es, ein vollständigeres Bild der Marktpositionierung zu geben und besser 
                        einzuschätzen, ob Preise noch weiter steigen oder fallen können. Besonders die PMPU-Gruppe 
                        (Produzenten und Konsumenten) liefert wertvolle Hinweise, da deren Hedging-Verhalten oft eine starke 
                        Verbindung zur physischen Marktlage hat.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Berechnung:**

                        x-Achse (Traderzahl):
                        $$
                        x \;=\; \#\;\text{MM Trader (Long oder Short)}
                        $$
                        - Anzahl der aktiven Money Manager Trader in Long- oder Short-Positionen

                        y-Achse (Positionsgrösse):
                        $$
                        y \;=\; \text{MM (Long oder Short) Open Interest}
                        $$
                        - gesamtes Open Interest (offene Kontrakte) der Money Manager in Long- oder Short-Positionen

                        Farbcodierung (Hedging-Kraft der PMPU):
                        $$
                        \text{Color}
                        \;=\;
                        \frac{\;\text{PMPU(L/S) OI}_{\text{current}} - \min\!\big(\text{PMPU(L/S) OI}_{\text{range}}\big)\;}
                             {\max\!\big(\text{PMPU(L/S) OI}_{\text{range}}\big) - \min\!\big(\text{PMPU(L/S) OI}_{\text{range}}\big)}
                        $$
                        - normiertes Open Interest der Produzenten/Verbraucher (PMPU), 
                          zeigt die aktuelle Position im Vergleich zu ihrem historischen Minimum und Maximum  
                        - **PMPU(L/S)** bezeichnet je nach Auswahl Long (PMPUL) oder Short (PMPUS)

                        **Weitere Visualisierungselemente:**
                        - **Grösse der Bubbles:** proportional zum gesamten Open Interest (Marktliquidität bzw. Marktgewicht)
                        - **Farbe der Bubbles:** zeigt die relative Stärke/Positionierung der PMPU-Gruppe im beobachteten Zeitraum
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.RadioItems(
            id='trader-group-radio',
            options=[
                {'label': 'MML', 'value': 'MML'},
                {'label': 'MMS', 'value': 'MMS'}
            ],
            value='MML',
            className='mb-4'
        ),
        dcc.Graph(id='hedging-indicator-graph')
    ], width=12)
]),

html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H2("DP Concentration / Clustering Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **DP Concentration / Clustering Indicator** kombiniert die Konzepte von Konzentration 
                        (Open Interest-Anteil) und Clustering (Anzahl Trader) in einem DP-Chart. Er zeigt, wie extrem die 
                        Positionierung einer Tradergruppe im Vergleich zu ihrer historischen Spanne ist.

                        Das **Ziel des Indikators** ist es, relative Handelschancen zwischen ähnlichen Märkten oder Rohstoffen 
                        aufzuzeigen, indem Positionierungsunterschiede sichtbar gemacht werden. Befinden sich z. B. beide 
                        Kennzahlen in einem Extrembereich, steigt die Wahrscheinlichkeit, dass ein Markt im Falle eines 
                        Preisschocks stärker reagiert als ein anderer.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Berechnung:**

                        **1) Clustering je Zeitpunkt $(t)$**

                        Rohanteil der Gruppe $(g)$ an allen Futures-Tradern:
                        $$
                        \mathrm{ClustShare}^{\mathrm{raw}}_g(m,t)=\frac{T_g(m,t)}{TT_F(m,t)}
                        $$
                        - Anteil der Trader der Gruppe \(g\) an der Gesamtzahl aller Futures-Trader im Markt
                        - Skaliert zwischen 0 und 1 (später normiert), unabhängig von der absoluten Traderzahl

                        Rolling-Normierung (1-Jahresfenster $\mathcal{W}_{365}$):
                        $$
                        \mathrm{ClustShare}^{\mathrm{roll}}_g(m,t)=
                        \frac{\mathrm{ClustShare}^{\mathrm{raw}}_g(m,t)-\min_{\tau\in\mathcal{W}_{365}}\mathrm{ClustShare}^{\mathrm{raw}}_g(m,\tau)}
                        {\max_{\tau\in\mathcal{W}_{365}}\mathrm{ClustShare}^{\mathrm{raw}}_g(m,\tau)-\min_{\tau\in\mathcal{W}_{365}}\mathrm{ClustShare}^{\mathrm{raw}}_g(m,\tau)}\cdot100
                        $$
                        - Setzt den aktuellen Rohanteil in Relation zur eigenen 1-Jahres-Historie dieses Marktes
                        - Ergebnis ist 0–100: 0 = historisches Jahres-Minimum, 100 = Jahres-Maximum

                        Zeitliche Aggregation im Fenster $[t_0,t_1]$:
                        $$
                        \overline{\mathrm{ClustShare}}^{\mathrm{roll}}_g(m)=
                        \frac{1}{|[t_0,t_1]|}\sum_{t=t_0}^{t_1}\mathrm{ClustShare}^{\mathrm{roll}}_g(m,t)
                        $$
                        - Glättet kurzfristiges Rauschen über das gewählte Analysefenster
                        - Liefert einen repräsentativen Durchschnittswert statt eines Einzelzeitpunkts

                        **2) Concentration je Zeitpunkt $(t)$**

                        Netto-Kontrakte (Long minus Short):
                        $$
                        \mathrm{RelConc}^{\mathrm{raw}}_g(m,t)=OI^{L}_g(m,t)-OI^{S}_g(m,t)
                        $$
                        - Misst die Richtung und Stärke der Positionierung der Gruppe $(g)$
                        - Positive Werte ⇒ Netto-Long; negative ⇒ Netto-Short

                        Zeitliche Aggregation im Fenster $[t_0,t_1]$:
                        $$
                        \overline{\mathrm{RelConc}}^{\mathrm{raw}}_g(m)=
                        \frac{1}{|[t_0,t_1]|}\sum_{t=t_0}^{t_1}\mathrm{RelConc}^{\mathrm{raw}}_g(m,t)
                        $$
                        - Mittelt die Netto-Position über das Analysefenster
                        - Reduziert Ausreisser, betont die **persistente** Positionierung

                        **3) Range-Normalisierung über alle Märkte (0–100)**

                        Clustering – Vergleichbarkeit über Märkte:
                        $$
                        \mathrm{ClusteringRange}_g(m)=
                        \frac{\overline{\mathrm{ClustShare}}^{\mathrm{roll}}_g(m)-\min_{m'}\overline{\mathrm{ClustShare}}^{\mathrm{roll}}_g(m')}
                        {\max_{m'}\overline{\mathrm{ClustShare}}^{\mathrm{roll}}_g(m')-\min_{m'}\overline{\mathrm{ClustShare}}^{\mathrm{roll}}_g(m')}\cdot100
                        $$
                        - Min-Max-Scaling quer über alle Märkte für die Cluster-Kennzahl
                        - 0 = niedrigster Markt im Sample, 100 = höchster ⇒ direkt vergleichbar

                        Concentration – Vergleichbarkeit über Märkte:
                        $$
                        \mathrm{ConcentrationRange}_g(m)=
                        \frac{\overline{\mathrm{RelConc}}^{\mathrm{raw}}_g(m)-\min_{m'}\overline{\mathrm{RelConc}}^{\mathrm{raw}}_g(m')}
                        {\max_{m'}\overline{\mathrm{RelConc}}^{\mathrm{raw}}_g(m')-\min_{m'}\overline{\mathrm{RelConc}}^{\mathrm{raw}}_g(m')}\cdot100
                        $$
                        - Min-Max-Scaling quer über alle Märkte für die Netto-Position
                        - Macht Märkte mit unterschiedlichen OI-Skalen vergleichbar auf 0–100

                        **4) Punkt im Plot (für Markt $(m)$)**
                        $$
                        x_m=\mathrm{ClusteringRange}_g(m),\qquad
                        y_m=\mathrm{ConcentrationRange}_g(m)
                        $$
                        - Jeder Markt wird ein Punkt $(x_m, y_m)$ im Scatter-Plot

                        **Interpretation:**
                        - **Clustering hoch ($x$ nahe 100)**: Im Vergleich zu Historie & anderen Märkten stark von Gruppe $g$ „gecrowded“
                        - **Concentration hoch ($y$ nahe 100)**: Markt zeigt (nach Zeitglättung) einen hohen Netto-Kontrakt-Überhang zugunsten der Gruppe $g$
                        - **Oben rechts** (hoch/hoch): doppelt extrem → Markt tendiert bei Schocks zu stärkeren Moves; **unten links**: unauffällig
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.DatePickerRange(
            id='concentration-clustering-date-picker-range',
            start_date=default_start_date,
            end_date=default_end_date,
            display_format='YYYY-MM-DD',
            className='mb-4'
        ),
        dcc.RadioItems(
            id='concentration-clustering-radio',
            options=[
                {'label': 'MML', 'value': 'MML'},
                {'label': 'MMS', 'value': 'MMS'}
            ],
            value='MML',
            inline=True,
            className='mb-4'
        ),
        dcc.Graph(id='dp-concentration-clustering-graph')
    ], width=12)
]),

html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("PP Concentration Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **Positioning Price Concentration Indicator** misst, wie gross der Anteil der Positionen 
                        einer Tradergruppe am gesamten Markt ist. Dadurch wird sichtbar, wie stark eine Gruppe auf 
                        der Long- oder Short-Seite im Verhältnis zum gesamten Open Interest vertreten ist. Eine hohe 
                        Konzentration deutet darauf hin, dass ein grosser Teil des Marktes von dieser Gruppe gehalten wird.

                        Das **Ziel des Indikators** ist es, die Marktbedeutung und Dominanz einer Tradergruppe sichtbar 
                        zu machen. Er zeigt, wie stark die Positionierung einer Gruppe relativ zum Gesamtmarkt ausfällt 
                        und hilft damit, Phasen hoher Konzentration zu erkennen. Dadurch lassen sich Rückschlüsse darauf 
                        ziehen, in welchen Marktphasen einzelne Gruppen besonders stark engagiert sind und wo potenzielle 
                        Risiken durch einen späteren Positionsabbau bestehen könnten.

                        **Farbskala:** Die Punktfarbe zeigt die Konzentration der Positionen der jeweiligen Gruppe am 
                        gesamten Open Interest.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Berechnung:**

                        $$
                        \mathrm{PP\ Concentration}_{G}(t)=
                        \frac{\mathrm{Position}_{G}(t)}
                        {\mathrm{OI}(t)} \times 100
                        $$
                        """, mathjax=True),

                        dcc.Markdown(r"""
                        **Bedeutung der Abkürzungen / Begriffe:**
                        - **MML:** Managed Money Long
                        - **MMS:** Managed Money Short
                        - **OI / Open Interest:** Gesamtzahl aller offenen Kontrakte im Markt
                        - **$\mathrm{Position}_{G}(t)$:** Position der betrachteten Gruppe am Reportdatum $t$, mit $G \in \{\mathrm{MML}, \mathrm{MMS}\}$
                        - **Concentration:** Anteil der Positionen einer Gruppe am gesamten Markt-OI
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.RadioItems(
            id='ppci-mm-radio',
            options=[
                {'label': 'Long', 'value': 'Long'},
                {'label': 'Short', 'value': 'Short'}
            ],
            value='Long',
            className='mb-4'
        ),

        dcc.Graph(id='positioning-price-concentration-graph'),
        html.Br(),
    ], width=12)
]),

        html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("PP Clustering Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **Positioning Price Clustering Indicator** misst, wie gross der Anteil der Trader 
                        einer bestimmten Tradergruppe an allen Long- bzw. Short-Tradern im Markt ist. Dadurch 
                        wird sichtbar, wie breit oder eng eine Positionierung innerhalb einer Gruppe abgestützt ist. 
                        Eine hohe Clustering-Ausprägung bedeutet, dass ein grosser Anteil der Marktteilnehmenden 
                        auf der jeweiligen Marktseite dieser Gruppe zuzuordnen ist.

                        Das **Ziel des Indikators** ist es, die Breite der Marktteilnahme einer Tradergruppe sichtbar 
                        zu machen. Er zeigt, ob eine Positionierung von vielen oder nur von wenigen Tradern getragen 
                        wird und macht damit die Struktur der Marktteilnahme transparenter. Dadurch lassen sich Phasen 
                        erkennen, in denen sich Positionierungen innerhalb einer Gruppe besonders stark verdichten oder verbreitern.

                        **Farbskala:** Die Punktfarbe zeigt das Clustering der jeweiligen Gruppe.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        **Berechnung:**

                        $$
                        \mathrm{PP\ Clustering}_{G}(t)=
                        \frac{\mathrm{Traders}_{G}(t)}
                        {\mathrm{Traders}^{\mathrm{side}}_{\mathrm{Total}}(t)} \times 100
                        $$
                        """, mathjax=True),

                        dcc.Markdown(r"""
                        **Bedeutung der Abkürzungen / Begriffe:**
                        - **MML:** Managed Money Long
                        - **MMS:** Managed Money Short
                        - **$G$:** betrachtete Gruppe mit $G \in \{\mathrm{MML}, \mathrm{MMS}\}$
                        - **$\mathrm{Traders}_{G}(t)$:** Anzahl Trader der betrachteten Gruppe am Reportdatum $t$
                        - **$\mathrm{Traders}^{\mathrm{side}}_{\mathrm{Total}}(t)$:** Gesamtzahl aller Trader derselben Marktseite (Long oder Short) am Reportdatum $t$
                        - **Clustering:** Anteil der Trader einer Gruppe an allen Tradern derselben Marktseite
                        """, mathjax=True),
                    ],
                    title="Berechnung",
                ),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.RadioItems(
            id='ppci-clustering-radio',
            options=[
                {'label': 'MML', 'value': 'MML'},
                {'label': 'MMS', 'value': 'MMS'}
            ],
            value='MML',
            className='mb-4'
        ),

        dcc.Graph(id='pp-clustering-graph'),
        html.Br(),
    ], width=12)
]),

html.Hr(),  # Separator
dbc.Row([
    dbc.Col([
        html.H1("PP Position Size Indicator"),

        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dcc.Markdown(r"""
                        Der **Positioning Price Position Size Indicator** misst die durchschnittliche Grösse 
                        der Positionen einzelner Trader innerhalb einer bestimmten Tradergruppe. Dazu wird die 
                        Position der betrachteten Gruppe mit dem Preis multipliziert und durch die Anzahl der 
                        Trader dieser Gruppe geteilt. Dadurch wird sichtbar, wie gross die durchschnittliche 
                        preisgewichtete Position pro Trader ist und wie stark einzelne Marktteilnehmende innerhalb 
                        der Gruppe engagiert sind.

                        Das **Ziel des Indikators** ist es, die durchschnittliche Positionsgrösse pro Trader und 
                        damit die Intensität des Engagements innerhalb einer Tradergruppe sichtbar zu machen. Er 
                        hilft zu beurteilen, ob Veränderungen im Open Interest eher durch eine steigende Zahl von 
                        Tradern oder durch grössere durchschnittliche Positionen pro Trader entstehen. Dadurch 
                        lassen sich Rückschlüsse auf die Stärke und Überzeugung der Marktteilnehmenden innerhalb 
                        einer Gruppe ziehen.

                        **Farbskala:** Die Punktfarbe zeigt die durchschnittliche Positionsgrösse der jeweiligen Gruppe.
                        """, mathjax=True),
                    ],
                    title="Beschreibung",
                ),

dbc.AccordionItem(
    [
        dcc.Markdown(r"""
        **Berechnung:**

        $$
        \mathrm{PP\ PositionSize}_{G}(t)=
        \frac{\mathrm{Position}_{G}(t)\times \mathrm{ContractSize}\times \mathrm{Price}(t)}
        {\mathrm{Traders}_{G}(t)}
        $$
        """, mathjax=True),

        dcc.Markdown(r"""
        **Bedeutung der Abkürzungen / Begriffe:**
        - **MML:** Managed Money Long
        - **MMS:** Managed Money Short
        - **$G$:** betrachtete Gruppe mit $G \in \{\mathrm{MML}, \mathrm{MMS}\}$
        - **$\mathrm{Position}_{G}(t)$:** Position der betrachteten Gruppe am Reportdatum $t$ (in Kontrakten)
        - **$\mathrm{ContractSize}$:** Kontraktgrösse des betrachteten Futures-Marktes
        - **$\mathrm{Price}(t)$:** Preis des betrachteten Futures-Marktes am Reportdatum $t$
        - **$\mathrm{Traders}_{G}(t)$:** Anzahl Trader der betrachteten Gruppe am Reportdatum $t$
        - **Position Size:** durchschnittliche preisgewichtete Positionsgrösse pro Trader innerhalb der betrachteten Gruppe
        """, mathjax=True),
    ],
    title="Berechnung",
),
            ],
            start_collapsed=True,
            always_open=True,
            flush=True,
            className="mb-4",
        ),

        dcc.RadioItems(
            id='ppci-position-size-radio',
            options=[
                {'label': 'MML', 'value': 'MML'},
                {'label': 'MMS', 'value': 'MMS'}
            ],
            value='MML',
            className='mb-4'
        ),

        dcc.Graph(id='pp-position-size-graph'),
        html.Br(),
    ], width=12)
]),

        html.Hr(),  # Separator
        dbc.Row([
            dbc.Col([
                html.Footer('© 2024 Market Analysis Dashboard', className='text-center mt-4')
            ])
        ])
    ], fluid=True)
])

def positions_bar(long_val, short_val, spread_val=None, bar_width_px=220, height_px=14):
    lv = 0 if pd.isna(long_val) else float(long_val)
    sv = 0 if pd.isna(short_val) else float(short_val)
    sp = 0 if (spread_val is None or pd.isna(spread_val)) else float(spread_val)

    lv = max(lv, 0)
    sv = max(sv, 0)
    sp = max(sp, 0)

    total = lv + sv + sp
    if total <= 0:
        return f"<div style='width:{bar_width_px}px;height:{height_px}px;border:1px solid #ccc;border-radius:3px;'></div>"

    p_long  = 100 * lv / total
    p_short = 100 * sv / total
    p_spread = 100 * sp / total

    spread_div = f"<div title='Spread: {int(sp)}' style='width:{p_spread:.2f}%;background:#1f77b4;'></div>" if sp > 0 else ""
    spread_txt = f", <b>Spread:</b> {int(sp)}" if sp > 0 else ""

    return (
        f"<div style='width:{bar_width_px}px;display:flex;flex-direction:column;'>"
        f"  <div style='display:flex;width:100%;height:{height_px}px;border:1px solid #ccc;border-radius:3px;overflow:hidden;'>"
        f"    <div title='Long: {int(lv)}'  style='width:{p_long:.2f}%;background:#2ca02c;'></div>"
        f"    <div title='Short: {int(sv)}' style='width:{p_short:.2f}%;background:#d62728;'></div>"
        f"    {spread_div}"
        f"  </div>"
        f"  <div style='font-size:11px;margin-top:4px;font-family:\"Courier New\", Courier, monospace;'>"
        f"    <b>Long:</b> {int(lv)}, <b>Short:</b> {int(sv)}{spread_txt}"
        f"  </div>"
        f"</div>"
    )

def traders_bar(long_val, short_val, spread_val=None, bar_width_px=220, height_px=14):
    lv = 0 if pd.isna(long_val) else float(long_val)
    sv = 0 if pd.isna(short_val) else float(short_val)
    tv = 0 if (spread_val is None or pd.isna(spread_val)) else float(spread_val)
    total = lv + sv + tv

    if total <= 0:
        return f"<div style='width:{bar_width_px}px;height:{height_px}px;border:1px solid #ccc;border-radius:3px;'></div>"

    p_long  = 100 * lv / total
    p_short = 100 * sv / total
    p_spread = 100 * tv / total

    spread_div = f"<div title='Spread: {int(tv)}' style='width:{p_spread:.2f}%;background:#1f77b4;'></div>" if tv > 0 else ""
    spread_txt = f", <b>Spread:</b> {int(tv)}" if tv > 0 else ""

    return (
        f"<div style='width:{bar_width_px}px;display:flex;flex-direction:column;'>"
        f"  <div style='display:flex;width:100%;height:{height_px}px;border:1px solid #ccc;border-radius:3px;overflow:hidden;'>"
        f"    <div title='Long: {int(lv)}'  style='width:{p_long:.2f}%;background:#2ca02c;'></div>"
        f"    <div title='Short: {int(sv)}' style='width:{p_short:.2f}%;background:#d62728;'></div>"
        f"    {spread_div}"
        f"  </div>"
        f"  <div style='font-size:11px;margin-top:4px;font-family:\"Courier New\", Courier, monospace;'>"
        f"    <b>Long:</b> {int(lv)}, <b>Short:</b> {int(sv)}{spread_txt}"
        f"  </div>"
        f"</div>"
    )

# Callback to update the table
@app.callback(
    Output('overview-table', 'data'),
    [
        Input('market-dropdown', 'value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date')
    ]
)
def update_table(selected_market, start_date, end_date):
    filtered_df = df_pivoted[
        (df_pivoted['Market Names'] == selected_market) &
        (df_pivoted['Date'] >= start_date) &
        (df_pivoted['Date'] <= end_date)
    ]

    if filtered_df.empty:
        return []

    first_row = filtered_df.iloc[0]
    current_row = filtered_df.iloc[-1]

    def safe_pct_change(curr, first):
        # verhindert Division durch 0 / NaN
        if pd.isna(first) or float(first) == 0:
            return 0
        return round(((float(curr) - float(first)) / float(first)) * 100, 2)

    data = {
        'Trader Group': [
            'Producer/Merchant/Processor/User',
            'Swap Dealer',
            'Managed Money',
            'Other Reportables'
        ],
        'Positions': [
            positions_bar(
                first_row['Producer/Merchant/Processor/User Long'],
                first_row['Producer/Merchant/Processor/User Short'],
                None
            ),
            positions_bar(
                first_row['Swap Dealer Long'],
                first_row['Swap Dealer Short'],
                first_row['Swap Dealer Spread']
            ),
            positions_bar(
                first_row['Managed Money Long'],
                first_row['Managed Money Short'],
                first_row['Managed Money Spread']
            ),
            positions_bar(
                first_row['Other Reportables Long'],
                first_row['Other Reportables Short'],
                first_row['Other Reportables Spread']
            ),
        ],

        'Difference (Long %)': [
            safe_pct_change(current_row['Producer/Merchant/Processor/User Long'], first_row['Producer/Merchant/Processor/User Long']),
            safe_pct_change(current_row['Swap Dealer Long'], first_row['Swap Dealer Long']),
            safe_pct_change(current_row['Managed Money Long'], first_row['Managed Money Long']),
            safe_pct_change(current_row['Other Reportables Long'], first_row['Other Reportables Long'])
        ],
        'Difference (Short %)': [
            safe_pct_change(current_row['Producer/Merchant/Processor/User Short'], first_row['Producer/Merchant/Processor/User Short']),
            safe_pct_change(current_row['Swap Dealer Short'], first_row['Swap Dealer Short']),
            safe_pct_change(current_row['Managed Money Short'], first_row['Managed Money Short']),
            safe_pct_change(current_row['Other Reportables Short'], first_row['Other Reportables Short'])
        ],
        'Difference (Spread %)': [
            0,  # PMPU hat bei dir keinen Spread
            safe_pct_change(current_row['Swap Dealer Spread'], first_row['Swap Dealer Spread']),
            safe_pct_change(current_row['Managed Money Spread'], first_row['Managed Money Spread']),
            safe_pct_change(current_row['Other Reportables Spread'], first_row['Other Reportables Spread'])
        ],

        'Total Traders': [
            current_row['Traders Prod/Merc Long'] + current_row['Traders Prod/Merc Short'],
            current_row['Traders Swap Long'] + current_row['Traders Swap Short'] + current_row['Traders Swap Spread'],
            current_row['Traders M Money Long'] + current_row['Traders M Money Short'] + current_row['Traders M Money Spread'],
            current_row['Traders Other Rept Long'] + current_row['Traders Other Rept Short'] + current_row['Traders Other Rept Spread']
        ],
        '% of Traders': [
            f"Long: {round(current_row['Traders Prod/Merc Long'] / current_row['Total Number of Traders'] * 100, 2)}%, "
            f"Short: {round(current_row['Traders Prod/Merc Short'] / current_row['Total Number of Traders'] * 100, 2)}%",

            f"Long: {round(current_row['Traders Swap Long'] / current_row['Total Number of Traders'] * 100, 2)}%, "
            f"Short: {round(current_row['Traders Swap Short'] / current_row['Total Number of Traders'] * 100, 2)}%, "
            f"Spread: {round(current_row['Traders Swap Spread'] / current_row['Total Number of Traders'] * 100, 2)}%",

            f"Long: {round(current_row['Traders M Money Long'] / current_row['Total Number of Traders'] * 100, 2)}%, "
            f"Short: {round(current_row['Traders M Money Short'] / current_row['Total Number of Traders'] * 100, 2)}%, "
            f"Spread: {round(current_row['Traders M Money Spread'] / current_row['Total Number of Traders'] * 100, 2)}%",

            f"Long: {round(current_row['Traders Other Rept Long'] / current_row['Total Number of Traders'] * 100, 2)}%, "
            f"Short: {round(current_row['Traders Other Rept Short'] / current_row['Total Number of Traders'] * 100, 2)}%, "
            f"Spread: {round(current_row['Traders Other Rept Spread'] / current_row['Total Number of Traders'] * 100, 2)}%"
        ],

        'Number of Traders': [
            traders_bar(current_row['Traders Prod/Merc Long'],  current_row['Traders Prod/Merc Short'],  None),
            traders_bar(current_row['Traders Swap Long'],       current_row['Traders Swap Short'],       current_row['Traders Swap Spread']),
            traders_bar(current_row['Traders M Money Long'],    current_row['Traders M Money Short'],    current_row['Traders M Money Spread']),
            traders_bar(current_row['Traders Other Rept Long'], current_row['Traders Other Rept Short'], current_row['Traders Other Rept Spread'])
        ],
    }

    return pd.DataFrame(data).to_dict('records')

# Callback to update graphs based on selected market and date range
@app.callback(
    [
        Output('long-clustering-graph', 'figure'),
        Output('short-clustering-graph', 'figure'),
        Output('pmpu-long-position-size-graph', 'figure'),
        Output('pmpu-short-position-size-graph', 'figure'),
        Output('sd-long-position-size-graph', 'figure'),
        Output('sd-short-position-size-graph', 'figure'),
        Output('long-position-size-graph', 'figure'),
        Output('short-position-size-graph', 'figure'),
        Output('or-long-position-size-graph', 'figure'),
        Output('or-short-position-size-graph', 'figure'),
        Output('dry-powder-indicator-graph', 'figure'),
        Output('dp-relative-concentration-graph', 'figure'),
        Output('dp-seasonal-indicator-graph', 'figure'),
        Output('dp-net-indicators-graph', 'figure'),
        Output('dp-position-size-indicator', 'figure'),
        Output('hedging-indicator-graph', 'figure')
    ],
    [Input('market-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('mm-radio', 'value'),
     Input('trader-group-radio', 'value')]
)

def update_graphs(selected_market, start_date, end_date, mm_type, trader_group):
    filtered_df = df_pivoted[(df_pivoted['Market Names'] == selected_market) &
                             (df_pivoted['Date'] >= start_date) & 
                             (df_pivoted['Date'] <= end_date)]

    # PMPU Long Position Size Indicator
    pmpu_long_position_size_fig = go.Figure()

    # Daten vorbereiten
    tr_long_raw = pd.to_numeric(filtered_df['Traders Prod/Merc Long'], errors='coerce')
    tr_long = tr_long_raw.fillna(0).clip(lower=0).astype(float)

    # Farbe = Positionsgröße (Long)
    try:
        col_long = safe_colors(filtered_df['PMPUL Position Size'])
    except Exception:
        col_long = pd.to_numeric(filtered_df['PMPUL Position Size'], errors='coerce').fillna(0)

    # 2) Durchmesser explizit auf Pixel mappen (z.B. 6–26 px)
    sizes_long = scaled_diameters(tr_long, min_px=6, max_px=26)

    # 3) Punkte plotten
    pmpu_long_position_size_fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Open Interest'],
            mode='markers',
            marker=dict(
                size=sizes_long,
                sizemode='diameter',
                sizeref=1,
                color=col_long,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="PMPU Long Position Size",
                    thickness=15, len=0.75, yanchor='middle', y=0.5
                )
            ),
            text=[
                f"Date: {d:%Y-%m-%d}<br>"
                f"Open Interest: {int(oi):,}<br>"
                f"Traders (Long): {int(t)}<br>"
                f"PosSize (avg): {float(ps):,.0f}"
                for d, oi, t, ps in zip(
                    filtered_df['Date'],
                    pd.to_numeric(filtered_df['Open Interest'], errors='coerce').fillna(0),
                    tr_long,
                    pd.to_numeric(filtered_df['PMPUL Position Size'], errors='coerce').fillna(0)
                )
            ],
            hoverinfo='text',
            showlegend=False
        )
    )

    # 4) Bubble-Size-Legende
    base = tr_long[tr_long > 0]
    if base.size >= 3 and base.max() > 1:
        legend_vals = np.unique(np.round(np.quantile(base, [0.25, 0.5, 0.75, 1.0])).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([10, 20, 35], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=6, max_px=26)

    for v, s in zip(legend_vals, legend_sizes):
        pmpu_long_position_size_fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
                showlegend=True,
                name=f"{int(v)} Traders",
                hoverinfo='skip'
            )
        )

    # 5) Layout
    pmpu_long_position_size_fig.update_layout(
        title='Long Position Size Indicator (PMPU)',
        xaxis_title='Date',
        yaxis_title='Open Interest',
        xaxis=dict(showgrid=True, ticks="outside", tickangle=45),
        yaxis=dict(title='Open Interest', showgrid=True),
        legend=dict(title=dict(text="Number of Traders"), itemsizing='trace', x=1.18, y=0.5, font=dict(size=12)),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    # 6) letzten Punkt hervorheben
    try:
        add_last_point_highlight(
            fig=pmpu_long_position_size_fig,
            df=filtered_df, x_col='Date', y_col='Open Interest',
            inner_size=2, inner_color='black'
        )
    except Exception:
        pass

    # PMPU Short Position Size Indicator
    pmpu_short_position_size_fig = go.Figure()

    # 1) Daten vorbereiten
    tr_short_raw = pd.to_numeric(filtered_df['Traders Prod/Merc Short'], errors='coerce')
    tr_short = tr_short_raw.fillna(0).clip(lower=0).astype(float)

    # Farbe = Positionsgröße (Short)
    try:
        col_short = safe_colors(filtered_df['PMPUS Position Size'])
    except Exception:
        col_short = pd.to_numeric(filtered_df['PMPUS Position Size'], errors='coerce').fillna(0)

    # 2) Durchmesser explizit auf Pixel mappen
    sizes_short = scaled_diameters(tr_short, min_px=6, max_px=26)

    # 3) Punkte plotten
    pmpu_short_position_size_fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Open Interest'],
            mode='markers',
            marker=dict(
                size=sizes_short,
                sizemode='diameter',
                sizeref=1,
                color=col_short,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="PMPU Short Position Size",
                    thickness=15, len=0.75, yanchor='middle', y=0.5
                )
            ),
            text=[
                f"Date: {d:%Y-%m-%d}<br>"
                f"Open Interest: {int(oi):,}<br>"
                f"Traders (Short): {int(t)}<br>"
                f"PosSize (avg): {float(ps):,.0f}"
                for d, oi, t, ps in zip(
                    filtered_df['Date'],
                    pd.to_numeric(filtered_df['Open Interest'], errors='coerce').fillna(0),
                    tr_short,
                    pd.to_numeric(filtered_df['PMPUS Position Size'], errors='coerce').fillna(0)
                )
            ],
            hoverinfo='text',
            showlegend=False
        )
    )

    # 4) Bubble-Size-Legende
    base_s = tr_short[tr_short > 0]
    if base_s.size >= 3 and base_s.max() > 1:
        legend_vals = np.unique(np.round(np.quantile(base_s, [0.25, 0.5, 0.75, 1.0])).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([10, 20, 35], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=6, max_px=26)

    for v, s in zip(legend_vals, legend_sizes):
        pmpu_short_position_size_fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
                showlegend=True,
                name=f"{int(v)} Traders",
                hoverinfo='skip'
            )
        )

    # 5) Layout
    pmpu_short_position_size_fig.update_layout(
        title='Short Position Size Indicator (PMPU)',
        xaxis_title='Date',
        yaxis_title='Open Interest',
        xaxis=dict(showgrid=True, ticks="outside", tickangle=45),
        yaxis=dict(title='Open Interest', showgrid=True),
        legend=dict(title=dict(text="Number of Traders"), itemsizing='trace', x=1.18, y=0.5, font=dict(size=12)),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    # 6) letzten Punkt hervorheben
    try:
        add_last_point_highlight(
            fig=pmpu_short_position_size_fig,
            df=filtered_df, x_col='Date', y_col='Open Interest',
            inner_size=2, inner_color='black'
        )
    except Exception:
        pass

    sd_long_position_size_fig = go.Figure()

    # 1) Daten vorbereiten
    sd_tr_long_raw = pd.to_numeric(filtered_df['Traders Swap Long'], errors='coerce')
    sd_tr_long = sd_tr_long_raw.fillna(0).clip(lower=0).astype(float)

    # Farbe = Positionsgröße (Long)
    try:
        sd_col_long = safe_colors(filtered_df['SDL Position Size'])
    except Exception:
        sd_col_long = pd.to_numeric(filtered_df['SDL Position Size'], errors='coerce').fillna(0)

    # 2) Durchmesser explizit auf Pixel mappen
    sd_sizes_long = scaled_diameters(sd_tr_long, min_px=6, max_px=26)

    # 3) Punkte plotten
    sd_long_position_size_fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Open Interest'],
            mode='markers',
            marker=dict(
                size=sd_sizes_long,
                sizemode='diameter',
                sizeref=1,
                color=sd_col_long,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="SD Long Position Size",
                    thickness=15, len=0.75, yanchor='middle', y=0.5
                )
            ),
            text=[
                f"Date: {d:%Y-%m-%d}<br>"
                f"Open Interest: {int(oi):,}<br>"
                f"Traders (Long): {int(t)}<br>"
                f"PosSize (avg): {float(ps):,.0f}"
                for d, oi, t, ps in zip(
                    filtered_df['Date'],
                    pd.to_numeric(filtered_df['Open Interest'], errors='coerce').fillna(0),
                    sd_tr_long,
                    pd.to_numeric(filtered_df['SDL Position Size'], errors='coerce').fillna(0)
                )
            ],
            hoverinfo='text',
            showlegend=False
        )
    )

    # 4) Bubble-Size-Legende (gleiche Skalierung wie oben)
    sd_baseL = sd_tr_long[sd_tr_long > 0]
    if sd_baseL.size >= 3 and sd_baseL.max() > 1:
        sd_legend_valsL = np.unique(np.round(np.quantile(sd_baseL, [0.25, 0.5, 0.75, 1.0])).astype(int))
        sd_legend_valsL = sd_legend_valsL[sd_legend_valsL > 0]
    else:
        sd_legend_valsL = np.array([10, 20, 35], dtype=int)

    sd_legend_sizesL = scaled_diameters(sd_legend_valsL, min_px=6, max_px=26)
    for v, s in zip(sd_legend_valsL, sd_legend_sizesL):
        sd_long_position_size_fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
                showlegend=True,
                name=f"{int(v)} Traders",
                hoverinfo='skip'
            )
        )

    # 5) Layout
    sd_long_position_size_fig.update_layout(
        title='Long Position Size Indicator (Swap Dealers)',
        xaxis_title='Date',
        yaxis_title='Open Interest',
        xaxis=dict(showgrid=True, ticks="outside", tickangle=45),
        yaxis=dict(title='Open Interest', showgrid=True),
        legend=dict(title=dict(text="Number of Traders"), itemsizing='trace', x=1.18, y=0.5, font=dict(size=12)),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    # 6) letzten Punkt hervorheben
    try:
        add_last_point_highlight(
            fig=sd_long_position_size_fig,
            df=filtered_df, x_col='Date', y_col='Open Interest',
            inner_size=2, inner_color='black'
        )
    except Exception:
        pass

    # SD Short Position Size Indicator
    sd_short_position_size_fig = go.Figure()

    # 1) Daten vorbereiten
    sd_tr_short_raw = pd.to_numeric(filtered_df['Traders Swap Short'], errors='coerce')
    sd_tr_short = sd_tr_short_raw.fillna(0).clip(lower=0).astype(float)

    # Farbe = Positionsgröße (Short)
    try:
        sd_col_short = safe_colors(filtered_df['SDS Position Size'])
    except Exception:
        sd_col_short = pd.to_numeric(filtered_df['SDS Position Size'], errors='coerce').fillna(0)

    # 2) Durchmesser explizit auf Pixel mappen
    sd_sizes_short = scaled_diameters(sd_tr_short, min_px=6, max_px=26)

    # 3) Punkte plotten
    sd_short_position_size_fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Open Interest'],
            mode='markers',
            marker=dict(
                size=sd_sizes_short,
                sizemode='diameter',
                sizeref=1,
                color=sd_col_short,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="SD Short Position Size",
                    thickness=15, len=0.75, yanchor='middle', y=0.5
                )
            ),
            text=[
                f"Date: {d:%Y-%m-%d}<br>"
                f"Open Interest: {int(oi):,}<br>"
                f"Traders (Short): {int(t)}<br>"
                f"PosSize (avg): {float(ps):,.0f}"
                for d, oi, t, ps in zip(
                    filtered_df['Date'],
                    pd.to_numeric(filtered_df['Open Interest'], errors='coerce').fillna(0),
                    sd_tr_short,
                    pd.to_numeric(filtered_df['SDS Position Size'], errors='coerce').fillna(0)
                )
            ],
            hoverinfo='text',
            showlegend=False
        )
    )

    # 4) Bubble-Size-Legende
    sd_baseS = sd_tr_short[sd_tr_short > 0]
    if sd_baseS.size >= 3 and sd_baseS.max() > 1:
        sd_legend_valsS = np.unique(np.round(np.quantile(sd_baseS, [0.25, 0.5, 0.75, 1.0])).astype(int))
        sd_legend_valsS = sd_legend_valsS[sd_legend_valsS > 0]
    else:
        sd_legend_valsS = np.array([10, 20, 35], dtype=int)

    sd_legend_sizesS = scaled_diameters(sd_legend_valsS, min_px=6, max_px=26)
    for v, s in zip(sd_legend_valsS, sd_legend_sizesS):
        sd_short_position_size_fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
                showlegend=True,
                name=f"{int(v)} Traders",
                hoverinfo='skip'
            )
        )

    # 5) Layout
    sd_short_position_size_fig.update_layout(
        title='Short Position Size Indicator (Swap Dealers)',
        xaxis_title='Date',
        yaxis_title='Open Interest',
        xaxis=dict(showgrid=True, ticks="outside", tickangle=45),
        yaxis=dict(title='Open Interest', showgrid=True),
        legend=dict(title=dict(text="Number of Traders"), itemsizing='trace', x=1.18, y=0.5, font=dict(size=12)),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    # 6) letzten Punkt hervorheben
    try:
        add_last_point_highlight(
            fig=sd_short_position_size_fig,
            df=filtered_df, x_col='Date', y_col='Open Interest',
            inner_size=2, inner_color='black'
        )
    except Exception:
        pass

    long_scaling_factor, short_scaling_factor = calculate_scaling_factors(filtered_df)

    # Long Positions Clustering
    long_clustering_fig = go.Figure()

    # 1) Trader-Serie sauber vorbereiten
    tr_total = pd.to_numeric(filtered_df['Total Number of Traders'], errors='coerce') \
        .fillna(0).clip(lower=0).astype(float)

    # 2) Marker-Grössen robust auf fixen Pixelbereich mappen
    MIN_PX = 8
    MAX_PX = 30
    sizes_total = scaled_diameters(tr_total, min_px=MIN_PX, max_px=MAX_PX)

    # 3) Scatter
    long_clustering_fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Open Interest'],
        mode='markers',
        marker=dict(
            size=sizes_total,
            sizemode='diameter',
            sizeref=1,
            color=filtered_df['Long Clustering'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Long Clustering (%)",
                thickness=15,
                len=0.75,
                yanchor='middle',
                y=0.5
            ),
        ),
        text=[
            f"Date: {d:%Y-%m-%d}<br>Traders: {int(t)}"
            for d, t in zip(filtered_df['Date'], tr_total)
        ],
        hoverinfo='text',
        showlegend=False
    ))

    # 4) Bubble-Size-Legende dynamisch (aus der Verteilung des selektierten Marktes)
    base = tr_total[tr_total > 0]
    if base.size >= 3:
        q = [0.10, 0.30, 0.50, 0.70, 0.90]  # 5 Legendenstufen
        legend_vals = np.unique(np.round(np.quantile(base, q)).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([50, 75, 100, 125, 150], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=MIN_PX, max_px=MAX_PX)

    for v, s in zip(legend_vals, legend_sizes):
        long_clustering_fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
            showlegend=True,
            name=f"{int(v)} Traders",
            hoverinfo='skip'
        ))

    # 5) Layout
    long_clustering_fig.update_layout(
        title='Long Positions Clustering Indicator',
        xaxis_title='Date',
        yaxis_title='Open Interest',
        xaxis=dict(
            tickmode='array',
            tickvals=filtered_df['Date'].dt.year.unique(),
            ticktext=[str(year) for year in filtered_df['Date'].dt.year.unique()],
            showgrid=True,
            ticks="outside",
            tickangle=45
        ),
        yaxis=dict(
            title='Open Interest',
            showgrid=True,
            tick0=0,
            dtick=20000 if selected_market in ['Gold', 'Silver', 'Copper'] else 5000,
        ),
        legend=dict(
            title=dict(text="Number of Traders"),
            itemsizing='trace',
            x=1.2,
            y=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    add_last_point_highlight(
        fig=long_clustering_fig,
        df=filtered_df,
        x_col='Date',
        y_col='Open Interest',
        inner_size=2,
        inner_color='black'
    )

    # Short Positions Clustering
    short_clustering_fig = go.Figure()

    # 1) Trader-Serie (gleich wie Long)
    tr_total = pd.to_numeric(filtered_df['Total Number of Traders'], errors='coerce') \
        .fillna(0).clip(lower=0).astype(float)

    # 2) gleiche Pixel-Skalierung
    MIN_PX = 8
    MAX_PX = 30
    sizes_total = scaled_diameters(tr_total, min_px=MIN_PX, max_px=MAX_PX)

    # 3) Scatter
    short_clustering_fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Open Interest'],
        mode='markers',
        marker=dict(
            size=sizes_total,
            sizemode='diameter',
            sizeref=1,
            color=filtered_df['Short Clustering'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Short Clustering (%)",
                thickness=15,
                len=0.75,
                yanchor='middle',
                y=0.5
            ),
        ),
        text=[
            f"Date: {d:%Y-%m-%d}<br>Traders: {int(t)}"
            for d, t in zip(filtered_df['Date'], tr_total)
        ],
        hoverinfo='text',
        showlegend=False
    ))

    # 4) Bubble-Size-Legende dynamisch
    base = tr_total[tr_total > 0]
    if base.size >= 3:
        q = [0.10, 0.30, 0.50, 0.70, 0.90]
        legend_vals = np.unique(np.round(np.quantile(base, q)).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([50, 75, 100, 125, 150], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=MIN_PX, max_px=MAX_PX)

    for v, s in zip(legend_vals, legend_sizes):
        short_clustering_fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
            showlegend=True,
            name=f"{int(v)} Traders",
            hoverinfo='skip'
        ))

    # 5) Layout
    short_clustering_fig.update_layout(
        title='Short Positions Clustering Indicator',
        xaxis_title='Date',
        yaxis_title='Open Interest',
        xaxis=dict(
            tickmode='array',
            tickvals=filtered_df['Date'].dt.year.unique(),
            ticktext=[str(year) for year in filtered_df['Date'].dt.year.unique()],
            showgrid=True,
            ticks="outside",
            tickangle=45
        ),
        yaxis=dict(
            title='Open Interest',
            showgrid=True,
            tick0=0,
            dtick=20000 if selected_market in ['Gold', 'Silver', 'Copper'] else 5000,
        ),
        legend=dict(
            title=dict(text="Number of Traders"),
            itemsizing='trace',
            x=1.2,
            y=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    add_last_point_highlight(
        fig=short_clustering_fig,
        df=filtered_df,
        x_col='Date',
        y_col='Open Interest',
        inner_size=2,
        inner_color='black'
    )

    # OR Long Position Size Indicator
    or_long_position_size_fig = go.Figure()

    # 1) Daten vorbereiten
    tr_long_raw  = pd.to_numeric(filtered_df['Traders Other Rept Long'], errors='coerce')
    tr_long = tr_long_raw.fillna(0).clip(lower=0).astype(float)

    # Farbe = Positionsgröße (Long)
    try:
        col_long = safe_colors(filtered_df['ORL Position Size'])
    except Exception:
        col_long = pd.to_numeric(filtered_df['ORL Position Size'], errors='coerce').fillna(0)

    # 2) Durchmesser explizit auf Pixel mappen
    sizes_long = scaled_diameters(tr_long, min_px=6, max_px=26)

    # 3) Punkte plotten
    or_long_position_size_fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Open Interest'],
            mode='markers',
            marker=dict(
                size=sizes_long,
                sizemode='diameter',
                sizeref=1,
                color=col_long,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="OR Long Position Size",
                    thickness=15, len=0.75, yanchor='middle', y=0.5
                )
            ),
            text=[
                f"Date: {d:%Y-%m-%d}<br>"
                f"Open Interest: {int(oi):,}<br>"
                f"Traders (Long): {int(t)}<br>"
                f"PosSize (avg): {float(ps):,.0f}"
                for d, oi, t, ps in zip(
                    filtered_df['Date'],
                    pd.to_numeric(filtered_df['Open Interest'], errors='coerce').fillna(0),
                    tr_long,
                    pd.to_numeric(filtered_df['ORL Position Size'], errors='coerce').fillna(0)
                )
            ],
            hoverinfo='text',
            showlegend=False
        )
    )

    # 4) Bubble-Size-Legende (gleiche Skalierung wie oben)
    base = tr_long[tr_long > 0]
    if base.size >= 3 and base.max() > 1:
        legend_vals = np.unique(np.round(np.quantile(base, [0.25, 0.5, 0.75, 1.0])).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([10, 20, 35], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=6, max_px=26)
    for v, s in zip(legend_vals, legend_sizes):
        or_long_position_size_fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
                showlegend=True,
                name=f"{int(v)} Traders",
                hoverinfo='skip'
            )
        )

    # 5) Layout
    or_long_position_size_fig.update_layout(
        title='Long Position Size Indicator (Other Reportables)',
        xaxis_title='Date',
        yaxis_title='Open Interest',
        xaxis=dict(showgrid=True, ticks="outside", tickangle=45),
        yaxis=dict(title='Open Interest', showgrid=True),
        legend=dict(title=dict(text="Number of Traders"), itemsizing='trace', x=1.18, y=0.5, font=dict(size=12)),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    # 6) letzten Punkt hervorheben
    try:
        add_last_point_highlight(
            fig=or_long_position_size_fig,
            df=filtered_df, x_col='Date', y_col='Open Interest',
            inner_size=2, inner_color='black'
        )
    except Exception:
        pass

    # OR Short Position Size Indicator
    or_short_position_size_fig = go.Figure()

    # 1) Daten vorbereiten
    tr_short_raw = pd.to_numeric(filtered_df['Traders Other Rept Short'], errors='coerce')
    tr_short = tr_short_raw.fillna(0).clip(lower=0).astype(float)

    # Farbe = Positionsgröße (Short)
    try:
        col_short = safe_colors(filtered_df['ORS Position Size'])
    except Exception:
        col_short = pd.to_numeric(filtered_df['ORS Position Size'], errors='coerce').fillna(0)

    # 2) Durchmesser explizit auf Pixel mappen
    sizes_short = scaled_diameters(tr_short, min_px=6, max_px=26)

    # 3) Punkte plotten
    or_short_position_size_fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Open Interest'],
            mode='markers',
            marker=dict(
                size=sizes_short,
                sizemode='diameter',
                sizeref=1,
                color=col_short,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="OR Short Position Size",
                    thickness=15, len=0.75, yanchor='middle', y=0.5
                )
            ),
            text=[
                f"Date: {d:%Y-%m-%d}<br>"
                f"Open Interest: {int(oi):,}<br>"
                f"Traders (Short): {int(t)}<br>"
                f"PosSize (avg): {float(ps):,.0f}"
                for d, oi, t, ps in zip(
                    filtered_df['Date'],
                    pd.to_numeric(filtered_df['Open Interest'], errors='coerce').fillna(0),
                    tr_short,
                    pd.to_numeric(filtered_df['ORS Position Size'], errors='coerce').fillna(0)
                )
            ],
            hoverinfo='text',
            showlegend=False
        )
    )

    # 4) Bubble-Size-Legende (gleiche Skalierung wie oben)
    base_s = tr_short[tr_short > 0]
    if base_s.size >= 3 and base_s.max() > 1:
        legend_vals = np.unique(np.round(np.quantile(base_s, [0.25, 0.5, 0.75, 1.0])).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([10, 20, 35], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=6, max_px=26)
    for v, s in zip(legend_vals, legend_sizes):
        or_short_position_size_fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
                showlegend=True,
                name=f"{int(v)} Traders",
                hoverinfo='skip'
            )
        )

    # 5) Layout
    or_short_position_size_fig.update_layout(
        title='Short Position Size Indicator (Other Reportables)',
        xaxis_title='Date',
        yaxis_title='Open Interest',
        xaxis=dict(showgrid=True, ticks="outside", tickangle=45),
        yaxis=dict(title='Open Interest', showgrid=True),
        legend=dict(title=dict(text="Number of Traders"), itemsizing='trace', x=1.18, y=0.5, font=dict(size=12)),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    # 6) letzten Punkt hervorheben
    try:
        add_last_point_highlight(
            fig=or_short_position_size_fig,
            df=filtered_df, x_col='Date', y_col='Open Interest',
            inner_size=2, inner_color='black'
        )
    except Exception:
        pass

    # --- MM Long Position Size Indicator ---
    long_position_size_fig = go.Figure()

    # 1) Daten vorbereiten (Größe = Anzahl Trader)
    mm_tr_long_raw = pd.to_numeric(filtered_df['Traders M Money Long'], errors='coerce')
    mm_tr_long = mm_tr_long_raw.fillna(0).clip(lower=0).astype(float)

    # Farbe = Positionsgröße (Long)
    try:
        mm_col_long = safe_colors(filtered_df['MML Position Size'])
    except Exception:
        mm_col_long = pd.to_numeric(filtered_df['MML Position Size'], errors='coerce').fillna(0)

    # 2) Durchmesser explizit auf Pixel mappen (z.B. 6–26 px)
    mm_sizes_long = scaled_diameters(mm_tr_long, min_px=6, max_px=26)

    # 3) Punkte plotten
    long_position_size_fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Open Interest'],
        mode='markers',
        marker=dict(
            size=mm_sizes_long,
            sizemode='diameter',
            sizeref=1,
            color=mm_col_long,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="MM Long Position Size", thickness=15, len=0.75, yanchor='middle', y=0.5)
        ),
        text=[
            f"Date: {d:%Y-%m-%d}<br>"
            f"Open Interest: {int(oi):,}<br>"
            f"Traders (Long): {int(t)}<br>"
            f"PosSize (avg): {float(ps):,.0f}"
            for d, oi, t, ps in zip(
                filtered_df['Date'],
                pd.to_numeric(filtered_df['Open Interest'], errors='coerce').fillna(0),
                mm_tr_long,
                pd.to_numeric(filtered_df['MML Position Size'], errors='coerce').fillna(0)
            )
        ],
        hoverinfo='text',
        showlegend=False
    ))

    # 4) Bubble-Size-Legende
    base = mm_tr_long[mm_tr_long > 0]
    if base.size >= 3 and base.max() > 1:
        legend_vals = np.unique(np.round(np.quantile(base, [0.25, 0.5, 0.75, 1.0])).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([10, 20, 35], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=6, max_px=26)
    for v, s in zip(legend_vals, legend_sizes):
        long_position_size_fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
            showlegend=True, name=f"{int(v)} Traders", hoverinfo='skip'
        ))

    # 5) Layout
    long_position_size_fig.update_layout(
        title='Long Position Size Indicator (Money Managers)',
        xaxis_title='Date', yaxis_title='Open Interest',
        xaxis=dict(
            tickmode='array',
            tickvals=filtered_df['Date'].dt.year.unique(),
            ticktext=[str(y) for y in filtered_df['Date'].dt.year.unique()],
            showgrid=True, ticks="outside", tickangle=45
        ),
        yaxis=dict(
            title='Open Interest', showgrid=True, tick0=0,
            dtick=20000 if selected_market in ['Gold', 'Silver', 'Copper'] else 5000
        ),
        legend=dict(title=dict(text="Number of Traders"), itemsizing='trace', x=1.2, y=0.5, font=dict(size=12)),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    # 6) letzten Punkt hervorheben
    try:
        add_last_point_highlight(long_position_size_fig, filtered_df, 'Date', 'Open Interest', inner_size=2,
                                 inner_color='black')
    except Exception:
        pass

    # --- MM Short Position Size Indicator ---
    short_position_size_fig = go.Figure()

    # 1) Daten vorbereiten (Größe = Anzahl Trader)
    mm_tr_short_raw = pd.to_numeric(filtered_df['Traders M Money Short'], errors='coerce')
    mm_tr_short = mm_tr_short_raw.fillna(0).clip(lower=0).astype(float)

    # Farbe = Positionsgröße (Short)
    try:
        mm_col_short = safe_colors(filtered_df['MMS Position Size'])
    except Exception:
        mm_col_short = pd.to_numeric(filtered_df['MMS Position Size'], errors='coerce').fillna(0)

    # 2) Durchmesser explizit auf Pixel mappen
    mm_sizes_short = scaled_diameters(mm_tr_short, min_px=6, max_px=26)

    # 3) Punkte plotten
    short_position_size_fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Open Interest'],
        mode='markers',
        marker=dict(
            size=mm_sizes_short,
            sizemode='diameter',
            sizeref=1,
            color=mm_col_short,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="MM Short Position Size", thickness=15, len=0.75, yanchor='middle', y=0.5)
        ),
        text=[
            f"Date: {d:%Y-%m-%d}<br>"
            f"Open Interest: {int(oi):,}<br>"
            f"Traders (Short): {int(t)}<br>"
            f"PosSize (avg): {float(ps):,.0f}"
            for d, oi, t, ps in zip(
                filtered_df['Date'],
                pd.to_numeric(filtered_df['Open Interest'], errors='coerce').fillna(0),
                mm_tr_short,
                pd.to_numeric(filtered_df['MMS Position Size'], errors='coerce').fillna(0)
            )
        ],
        hoverinfo='text',
        showlegend=False
    ))

    # 4) Bubble-Size-Legende
    base_s = mm_tr_short[mm_tr_short > 0]
    if base_s.size >= 3 and base_s.max() > 1:
        legend_vals = np.unique(np.round(np.quantile(base_s, [0.25, 0.5, 0.75, 1.0])).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([10, 20, 35], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=6, max_px=26)
    for v, s in zip(legend_vals, legend_sizes):
        short_position_size_fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=float(s), sizemode='diameter', sizeref=1, color='gray', opacity=0.6),
            showlegend=True, name=f"{int(v)} Traders", hoverinfo='skip'
        ))

    # 5) Layout
    short_position_size_fig.update_layout(
        title='Short Position Size Indicator (Money Managers)',
        xaxis_title='Date', yaxis_title='Open Interest',
        xaxis=dict(
            tickmode='array',
            tickvals=filtered_df['Date'].dt.year.unique(),
            ticktext=[str(y) for y in filtered_df['Date'].dt.year.unique()],
            showgrid=True, ticks="outside", tickangle=45
        ),
        yaxis=dict(
            title='Open Interest', showgrid=True, tick0=0,
            dtick=20000 if selected_market in ['Gold', 'Silver', 'Copper'] else 5000
        ),
        legend=dict(title=dict(text="Number of Traders"), itemsizing='trace', x=1.2, y=0.5, font=dict(size=12)),
        margin=dict(l=60, r=160, t=60, b=60)
    )

    # 6) letzten Punkt hervorheben
    try:
        add_last_point_highlight(short_position_size_fig, filtered_df, 'Date', 'Open Interest', inner_size=2,
                                 inner_color='black')
    except Exception:
        pass

    # --- Dry Powder Indicator---
    dry_powder_fig = go.Figure()

    bubble_size = (filtered_df['MML Long OI'].abs() + filtered_df['MML Short OI'].abs())
    desired_max_px = 28
    sizeref = 2.0 * bubble_size.max() / (desired_max_px ** 2)

    COL_LONG = "#2c7fb8"  # MML
    COL_SHORT = "#7fcdbb"  # MMS

    # MML Wolke
    dry_powder_fig.add_trace(go.Scatter(
        x=filtered_df['MML Traders'],
        y=filtered_df['MML Long OI'],
        mode='markers',
        marker=dict(
            size=bubble_size, sizemode='area', sizeref=sizeref,
            color=COL_LONG, opacity=0.75, line=dict(width=0.6, color='black')
        ),
        name='MML'
    ))

    # MMS Wolke
    dry_powder_fig.add_trace(go.Scatter(
        x=filtered_df['MMS Traders'],
        y=filtered_df['MML Short OI'],
        mode='markers',
        marker=dict(
            size=bubble_size, sizemode='area', sizeref=sizeref,
            color=COL_SHORT, opacity=0.75, line=dict(width=0.6, color='black')
        ),
        name='MMS'
    ))

    # x-Range über beide Gruppen
    x_min = float(min(filtered_df['MML Traders'].min(), filtered_df['MMS Traders'].min()))
    x_max = float(max(filtered_df['MML Traders'].max(), filtered_df['MMS Traders'].max()))
    xs = np.array([x_min, x_max])

    def add_trend(x_series, y_series, color, name):
        # NaNs entfernen
        mask = x_series.notna() & y_series.notna()
        x = x_series[mask].astype(float).values
        y = y_series[mask].astype(float).values
        if len(x) < 2:
            return
        m, b = np.polyfit(x, y, 1)
        ys = m * xs + b

        # Unterzug (weiß, breit) für bessere Sichtbarkeit
        dry_powder_fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color='white', width=7),
            name=name, showlegend=False, hoverinfo='skip'
        ))
        # Farblinie oben drauf
        dry_powder_fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=3),
            name=name, showlegend=True
        ))

    # Trendlinien hinzufügen (durch den ganzen Graph)
    add_trend(filtered_df['MML Traders'], filtered_df['MML Long OI'], COL_LONG, "MML Trend")
    add_trend(filtered_df['MMS Traders'], filtered_df['MML Short OI'], COL_SHORT, "MMS Trend")

    # Most Recent Week
    dry_powder_fig.add_trace(go.Scatter(
        x=[filtered_df['MML Traders'].iloc[-1]],
        y=[filtered_df['MML Long OI'].iloc[-1]],
        mode='markers',
        marker=dict(size=desired_max_px + 4, color='black', line=dict(width=2, color='white')),
        name='Most Recent Week', legendgroup='recent', showlegend=True
    ))
    dry_powder_fig.add_trace(go.Scatter(
        x=[filtered_df['MMS Traders'].iloc[-1]],
        y=[filtered_df['MML Short OI'].iloc[-1]],
        mode='markers',
        marker=dict(size=desired_max_px + 4, color='black', line=dict(width=2, color='white')),
        name='Most Recent Week', legendgroup='recent', showlegend=False  # keine doppelte Legende
    ))

    dry_powder_fig.update_layout(
        title=f"Dry Powder Indicator",
        xaxis=dict(title='Number of Traders', showgrid=True, gridcolor='LightGray', gridwidth=2, zeroline=False),
        yaxis=dict(title='Long and Short OI', showgrid=True, gridcolor='LightGray', gridwidth=2, zeroline=False),
        plot_bgcolor='white',
        legend_title='Trader Group'
    )

    # --- DP Relative Concentration Indicator (konstante Grösse + 8 schwarze Punkte) ---
    fig_rc = go.Figure()

    TOTAL_OI = pd.to_numeric(filtered_df.get('Open Interest'), errors='coerce').replace(0, np.nan)

    def rc(long_col, short_col):
        L = pd.to_numeric(filtered_df.get(long_col), errors='coerce')
        S = pd.to_numeric(filtered_df.get(short_col), errors='coerce')
        return 100.0 * ((L / TOTAL_OI) - (S / TOTAL_OI))

    groups = [
        dict(name='MML', x='Traders M Money Long',
             rc=rc('Managed Money Long', 'Managed Money Short'), color='#2c7fb8'),
        dict(name='MMS', x='Traders M Money Short',
             rc=rc('Managed Money Short', 'Managed Money Long'), color='#7fcdbb'),
        dict(name='ORL', x='Traders Other Rept Long',
             rc=rc('Other Reportables Long', 'Other Reportables Short'), color='#f39c12'),
        dict(name='ORS', x='Traders Other Rept Short',
             rc=rc('Other Reportables Short', 'Other Reportables Long'), color='#f1c40f'),
        dict(name='PMPUL', x='Traders Prod/Merc Long',
             rc=rc('Producer/Merchant/Processor/User Long',
                   'Producer/Merchant/Processor/User Short'), color='#27ae60'),
        dict(name='PMPUS', x='Traders Prod/Merc Short',
             rc=rc('Producer/Merchant/Processor/User Short',
                   'Producer/Merchant/Processor/User Long'), color='#2ecc71'),
        dict(name='SDL', x='Traders Swap Long',
             rc=rc('Swap Dealer Long', 'Swap Dealer Short'), color='#e67e22'),
        dict(name='SDS', x='Traders Swap Short',
             rc=rc('Swap Dealer Short', 'Swap Dealer Long'), color='#e74c3c'),
    ]

    # 1) KONSTANTE Bubble-Grösse in Pixel (für alle gleich)
    bubble_px = 14
    recent_px = bubble_px + 6

    # Historische Punkte je Gruppe
    for g in groups:
        x = pd.to_numeric(filtered_df.get(g['x']), errors='coerce')
        y = g['rc']
        mask = x.notna() & y.notna()
        if mask.sum() == 0:
            continue

        fig_rc.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            mode='markers',
            marker=dict(
                size=bubble_px,
                color=g['color'],
                opacity=0.8,
                line=dict(width=0.6, color='black')
            ),
            name=g['name']
        ))

    # 2) PRO GRUPPE: schwarzer Punkt für die letzte verfügbare Beobachtung
    first_legend_done = False
    for g in groups:
        x_series = pd.to_numeric(filtered_df.get(g['x']), errors='coerce')
        y_series = g['rc']
        mask = x_series.notna() & y_series.notna()
        if mask.sum() == 0:
            continue
        last_idx = y_series[mask].index[-1]
        x_last = x_series.loc[last_idx]
        y_last = y_series.loc[last_idx]
        if pd.notna(x_last) and pd.notna(y_last):
            fig_rc.add_trace(go.Scatter(
                x=[x_last], y=[y_last],
                mode='markers',
                marker=dict(size=recent_px, color='black', line=dict(width=2, color='white')),
                name='Most Recent Week',
                legendgroup='recent',
                showlegend=not first_legend_done  # nur 1 Legenden-Eintrag
            ))
            first_legend_done = True

    fig_rc.update_layout(
        title="DP Relative Concentration Indicator",
        xaxis=dict(title='Number of Traders', showgrid=True, gridcolor='LightGray'),
        yaxis=dict(title='Long and Short Concentration', showgrid=True, gridcolor='LightGray'),
        plot_bgcolor='white',
        legend_title='Trader Group'
    )

    # DP Seasonal Indicator
    dp_seasonal_indicator_fig = go.Figure()

    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    colors = ['#1f77b4', '#17becf', '#ff7f0e', '#d62728']

    for quarter, color in zip(quarters, colors):
        quarter_data = filtered_df[filtered_df['Quarter'] == quarter]
        if quarter_data.empty:
            continue

        dp_seasonal_indicator_fig.add_trace(go.Scatter(
            x=quarter_data['PMPUL Traders'],
            y=quarter_data['PMPUL Relative Concentration'],
            mode='markers',
            marker=dict(
                size=10,  # 🔹 Fixe, einheitliche Bubblegröße
                color=color,
                opacity=0.7,
                line=dict(width=0.6, color='black')
            ),
            name=quarter
        ))

    # Schwarzer Punkt für Most Recent Week
    most_recent_date = filtered_df['Date'].max()
    recent_data = filtered_df[filtered_df['Date'] == most_recent_date]
    if not recent_data.empty:
        dp_seasonal_indicator_fig.add_trace(go.Scatter(
            x=recent_data['PMPUL Traders'],
            y=recent_data['PMPUL Relative Concentration'],
            mode='markers',
            marker=dict(
                size=12,  # etwas grösser zur Hervorhebung
                color='black',
                symbol='circle',
                line=dict(width=1.5, color='white')
            ),
            name='Most Recent Week'
        ))

    dp_seasonal_indicator_fig.update_layout(
        title=f"DP Seasonal Indicator – {most_recent_date.strftime('%d/%m/%Y')}",
        xaxis_title="Number of Traders",
        yaxis_title="Long and Short Concentration",
        plot_bgcolor='white',
        legend_title="Quarter",
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray')
    )

    # DP Net Indicators with Medians
    most_recent_date = filtered_df['Date'].max()
    first_date = filtered_df['Date'].min()
    median_oi, median_traders = calculate_medians(filtered_df)
    
    dp_net_indicators_fig = go.Figure()

    # Color coding by Year
    for year in filtered_df['Year'].unique():
        year_data = filtered_df[filtered_df['Year'] == year]

        dp_net_indicators_fig.add_trace(go.Scatter(
            x=year_data['MM Net Traders'],
            y=year_data['MM Net OI'],
            mode='markers',
            marker=dict(size=10, opacity=0.6),
            name=str(year)
        ))

    # Adding markers for the most recent and first weeks
    recent_data = filtered_df[filtered_df['Date'] == most_recent_date]
    first_data = filtered_df[filtered_df['Date'] == first_date]
    
    dp_net_indicators_fig.add_trace(go.Scatter(
        x=recent_data['MM Net Traders'],
        y=recent_data['MM Net OI'],
        mode='markers',
        marker=dict(size=12, color='black', symbol='circle'),
        name='Most Recent Week'
    ))

    dp_net_indicators_fig.add_trace(go.Scatter(
        x=first_data['MM Net Traders'],
        y=first_data['MM Net OI'],
        mode='markers',
        marker=dict(size=12, color='red', symbol='circle'),
        name='First Week'
    ))

    # Adding medians
    dp_net_indicators_fig.add_trace(go.Scatter(
        x=[median_traders, median_traders],
        y=[filtered_df['MM Net OI'].min(), filtered_df['MM Net OI'].max()],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Median Net Traders'
    ))

    dp_net_indicators_fig.add_trace(go.Scatter(
        x=[filtered_df['MM Net Traders'].min(), filtered_df['MM Net Traders'].max()],
        y=[median_oi, median_oi],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Median Net OI'
    ))

    dp_net_indicators_fig.update_layout(
        title='DP Net Indicators with Medians',
        xaxis_title='MM Net Number of Traders',
        yaxis_title='MM Net OI',
        legend_title='Year'
    )

    # Dry Powder Position Size Indicator (MML & MMS)
    dff = filtered_df
    if mm_type == 'MML':
        x = dff['Traders M Money Long']
        y = dff['MML Position Size']
        color = dff['Open Interest']
        recent_week = dff['MML Position Size'].iloc[-1]
        recent_x = dff['Traders M Money Long'].iloc[-1]
        first_week = dff['MML Position Size'].iloc[0]
        first_x = dff['Traders M Money Long'].iloc[0]
    else:
        x = dff['Traders M Money Short']
        y = dff['MMS Position Size']
        color = dff['Open Interest']
        recent_week = dff['MMS Position Size'].iloc[-1]
        recent_x = dff['Traders M Money Short'].iloc[-1]
        first_week = dff['MMS Position Size'].iloc[0]
        first_x = dff['Traders M Money Short'].iloc[0]

    median_x = x.median()
    median_y = y.median()

    dp_position_size_fig = go.Figure()

    dp_position_size_fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=10,
            color=color,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Open Interest',
                thickness=15,
                len=0.75,
                yanchor='middle'
            )
        ),
        text=dff['Date'],
        hoverinfo='text',
        showlegend=False
    ))

    dp_position_size_fig.add_trace(go.Scatter(
        x=[recent_x],
        y=[recent_week],
        mode='markers',
        marker=dict(
            size=12,
            color='black'
        ),
        name='Most Recent Week'
    ))

    dp_position_size_fig.add_trace(go.Scatter(
        x=[first_x],
        y=[first_week],
        mode='markers',
        marker=dict(
            size=12,
            color='red'
        ),
        name='First Week'
    ))

    dp_position_size_fig.add_shape(type="line",
                  x0=median_x, y0=0, x1=median_x, y1=max(y),
                  line=dict(color="Gray", width=1, dash="dash"))

    dp_position_size_fig.add_shape(type="line",
                  x0=0, y0=median_y, x1=max(x), y1=median_y,
                  line=dict(color="Gray", width=1, dash="dash"))

    dp_position_size_fig.update_layout(
        title='Dry Powder Position Size Indicator ({})'.format(mm_type),
        xaxis_title='Number of {} Traders'.format(mm_type),
        yaxis_title='{} Position Size'.format(mm_type),
        showlegend=True,
    )

    # Dry Powder Hedging Indicator (MML vs PMPUL / MMS vs PMPUS)
    hedging_fig = create_hedging_indicator(filtered_df, trader_group, start_date, end_date)

    return (
        long_clustering_fig,
        short_clustering_fig,
        pmpu_long_position_size_fig,
        pmpu_short_position_size_fig,
        sd_long_position_size_fig,
        sd_short_position_size_fig,
        long_position_size_fig,
        short_position_size_fig,
        or_long_position_size_fig,
        or_short_position_size_fig,
        dry_powder_fig,
        fig_rc,
        dp_seasonal_indicator_fig,
        dp_net_indicators_fig,
        dp_position_size_fig,
        hedging_fig
    )


# Function to create the hedging indicator
def create_hedging_indicator(data, trader_group, start_date, end_date):
    import numpy as np
    # Filter data by date range
    mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
    data = data.loc[mask]

    if trader_group == "MML":
        x = 'Traders M Money Long'
        y = 'MML Long OI'
        color = 'PMPUL Relative Concentration'
        title = 'Dry Powder Hedging Indicator (MML vs PMPUL)'
        colorbar_title = 'PMPUL OI Range'
        x_title = 'MM Number of Long Traders'
        y_title = 'MM Long OI'
    else:
        x = 'Traders M Money Short'
        y = 'MMS Short OI'
        color = 'PMPUS Relative Concentration'
        title = 'Dry Powder Hedging Indicator (MMS vs PMPUS)'
        colorbar_title = 'PMPUS OI Range'
        x_title = 'MM Number of Short Traders'
        y_title = 'MM Short OI'

    # Vorab die gewünschten Achsenranges bestimmen (benötigen wir auch für die Trendlinie)
    x_min = float(np.nanmin(data[x])) - 10
    x_max = float(np.nanmax(data[x])) + 10
    y_min = float(np.nanmin(data[y])) - 50000
    y_max = float(np.nanmax(data[y])) + 50000

    # Haupt-Scatter
    # --- Bubble sizing---
    oi = pd.to_numeric(data['Open Interest'], errors='coerce').abs()

    desired_max_px = 26
    desired_min_px = 6
    sizeref = 2.0 * oi.max() / (desired_max_px ** 2)

    trace = go.Scatter(
        x=data[x],
        y=data[y],
        mode='markers',
        marker=dict(
            size=oi,
            sizemode='area',
            sizeref=sizeref,
            sizemin=desired_min_px,
            color=data[color],
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title=colorbar_title, len=0.5, x=1.1)
        ),
        text=data['Market Names'],
        hoverinfo='text',
        showlegend=False
    )

    # First / Last Week Marker
    first_week = data.iloc[0]
    last_week = data.iloc[-1]

    first_week_trace = go.Scatter(
        x=[first_week[x]], y=[first_week[y]],
        mode='markers', marker=dict(color='red', size=15),
        name='First Week'
    )
    last_week_trace = go.Scatter(
        x=[last_week[x]], y=[last_week[y]],
        mode='markers', marker=dict(color='black', size=15),
        name='Most Recent Week'
    )

    # --- Trendlinie (OLS) - über die ganze Plotbreite, solid, ohne Legende/Annotation ---
    xv = data[x].astype(float).to_numpy()
    yv = data[y].astype(float).to_numpy()
    mask_finite = np.isfinite(xv) & np.isfinite(yv)

    trend_trace = None
    if mask_finite.sum() >= 2:
        m, c = np.polyfit(xv[mask_finite], yv[mask_finite], 1)
        x_line = np.array([x_min, x_max])
        y_line = m * x_line + c
        trend_trace = go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='skip',
            showlegend=False
        )

    # Layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title=x_title, range=[x_min, x_max]),
        yaxis=dict(title=y_title, range=[y_min, y_max]),
        showlegend=True,
        width=1000, height=600
    )

    # Figure zusammensetzen
    traces = [trace, first_week_trace, last_week_trace]
    if trend_trace:
        traces.append(trend_trace)

    fig = go.Figure(data=traces, layout=layout)
    return fig

# Callback to update the Dry Powder Concentration/Clustering Indicator graph
@app.callback(
    Output('dp-concentration-clustering-graph', 'figure'),
    [Input('concentration-clustering-date-picker-range', 'start_date'),
     Input('concentration-clustering-date-picker-range', 'end_date'),
     Input('concentration-clustering-radio', 'value')]
)
def update_concentration_clustering_graph(start_date, end_date, selected_indicator):
    filtered_df = df_pivoted[(df_pivoted['Date'] >= start_date) & 
                             (df_pivoted['Date'] <= end_date)]
    
    # Aggregate the data by market, keeping only numeric columns
    agg_df = filtered_df.groupby('Market Names').mean(numeric_only=True).reset_index()
    
    concentration_range, clustering_range = calculate_ranges(agg_df, selected_indicator)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=clustering_range,
        y=concentration_range,
        mode='markers+text',
        text=agg_df['Market Names'],
        textposition='top center',
        marker=dict(size=10, opacity=0.6, color='green', line=dict(width=1, color='black'))
    ))
    
    fig.update_layout(
        title=f'Dry Powder Concentration/Clustering Indicator ({selected_indicator})',
        xaxis_title='MM Long Clustering Range' if selected_indicator == 'MML' else 'MM Short Clustering Range',
        yaxis_title='MM Long Concentration Range' if selected_indicator == 'MML' else 'MM Short Concentration Range',
        xaxis=dict(range=[-5, 110]),  # Adjusted to ensure all bubbles are visible
        yaxis=dict(range=[-5, 110]),  # Adjusted to ensure all bubbles are visible
        showlegend=False
    )
    
    return fig


# ---------------------------------------------------------------------------
# PPCI – Callback
# ---------------------------------------------------------------------------
@app.callback(
    Output('positioning-price-concentration-graph', 'figure'),
    [Input('market-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('ppci-mm-radio', 'value')]
)
def update_ppci(selected_market, start_date, end_date, direction):
    dff = df_pivoted[
        (df_pivoted['Market Names'] == selected_market) &
        (df_pivoted['Date'] >= start_date) &
        (df_pivoted['Date'] <= end_date)
    ].copy()

    if dff.empty:
        return go.Figure()

    # Normalize Date to tz-naive for merging
    dff['_date'] = pd.to_datetime(dff['Date']).dt.tz_localize(None)

    # Long/Short Concentration (%)
    total_oi = pd.to_numeric(dff['Open Interest'], errors='coerce').replace(0, np.nan)
    dff['_long_conc']  = 100.0 * pd.to_numeric(dff['Managed Money Long'],  errors='coerce') / total_oi
    dff['_short_conc'] = 100.0 * pd.to_numeric(dff['Managed Money Short'], errors='coerce') / total_oi

    if direction == 'Long':
        color_col      = '_long_conc'
        colorbar_title = 'Long Concentration (%)'
    else:
        color_col      = '_short_conc'
        colorbar_title = 'Short Concentration (%)'

    # Merge futures prices from InfluxDB (continuous contract proxy for 2nd Nearby)
    price_col = _ppci_get_price_col(selected_market)
    y_title = 'Price (Continuous Front-Month Proxy) (Report Date)'

    if price_col and not df_futures_prices.empty and price_col in df_futures_prices.columns:
        prices = df_futures_prices[['Date', price_col]].dropna(subset=[price_col]).copy()
        prices = prices.rename(columns={'Date': '_pdate'}).sort_values('_pdate')
        dff = dff.sort_values('_date')

        dff = pd.merge_asof(
            dff, prices,
            left_on='_date', right_on='_pdate',
            direction='backward',
            tolerance=pd.Timedelta(days=7)
        )
        y_vals = pd.to_numeric(dff[price_col], errors='coerce')
    else:
        y_vals = pd.Series([np.nan] * len(dff), index=dff.index)
        y_title = f'Price (keine Daten für {selected_market})'

    # Bubble sizing: Total Open Interest
    oi = pd.to_numeric(dff['Open Interest'], errors='coerce').fillna(0).abs()

    MIN_PX = 8
    MAX_PX = 30
    sizes_oi = scaled_diameters(oi, min_px=MIN_PX, max_px=MAX_PX)

    color_vals = pd.to_numeric(dff[color_col], errors='coerce')

    # Hover-Text
    dates_str = pd.to_datetime(dff['Date']).dt.strftime('%Y-%m-%d')
    hover_text = [
        f"Date: {d}<br>Price: {p:.2f}<br>Total Open Interest: {o:,.0f}<br>{colorbar_title}: {c:.1f}%"
        for d, p, o, c in zip(
            dates_str,
            y_vals.fillna(0),
            oi.fillna(0),
            color_vals.fillna(0)
        )
    ]

    fig = go.Figure()

    # Haupttrace
    fig.add_trace(go.Scatter(
        x=dff['Date'],
        y=y_vals,
        mode='markers',
        marker=dict(
            size=sizes_oi,
            sizemode='diameter',
            sizeref=1,
            color=color_vals,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(
                title=colorbar_title,
                thickness=15,
                len=0.75,
                yanchor='middle',
                y=0.5
            ),
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))

    # Bubble-Size-Legende dynamisch
    base = oi[oi > 0]
    if base.size >= 3:
        q = [0.10, 0.30, 0.50, 0.70, 0.90]
        legend_vals = np.unique(np.round(np.quantile(base, q)).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([50000, 100000, 150000, 200000, 250000], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=MIN_PX, max_px=MAX_PX)

    for v, s in zip(legend_vals, legend_sizes):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=float(s),
                sizemode='diameter',
                sizeref=1,
                color='gray',
                opacity=0.6
            ),
            showlegend=True,
            name=f"{int(v):,}",
            hoverinfo='skip'
        ))

    # Letzter Datenpunkt hervorheben
    if not dff.empty and not y_vals.empty:
        fig.add_trace(go.Scatter(
            x=[dff.iloc[-1]['Date']],
            y=[y_vals.iloc[-1]],
            mode='markers',
            marker=dict(
                size=10,
                color='black',
                opacity=1.0,
                line=dict(width=4, color='red')
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f'Positioning Price Concentration Indicator ({direction}) – {selected_market}',
        xaxis_title='Report Date',
        yaxis_title=y_title,
        legend=dict(
            title=dict(text='Total Open Interest'),
            itemsizing='trace',
            x=1.2,
            y=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=160, t=60, b=60),
        height=600,
    )

    return fig


# ---------------------------------------------------------------------------
# PP Clustering Indicator – Callback
# Wiederverwendet: _ppci_get_price_col, df_futures_prices, merge_asof,
# scaled_diameters, OI-Legende und Letzter-Punkt-Highlight aus dem PPCI.
# Unterschied zum PPCI: Farbe = Long/Short Clustering statt Concentration (%).
# ---------------------------------------------------------------------------
@app.callback(
    Output('pp-clustering-graph', 'figure'),
    [Input('market-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('ppci-clustering-radio', 'value')]
)
def update_pp_clustering(selected_market, start_date, end_date, mm_type):
    dff = df_pivoted[
        (df_pivoted['Market Names'] == selected_market) &
        (df_pivoted['Date'] >= start_date) &
        (df_pivoted['Date'] <= end_date)
    ].copy()

    if dff.empty:
        return go.Figure()

    # Normalize Date to tz-naive for merging (identisch zu PPCI)
    dff['_date'] = pd.to_datetime(dff['Date']).dt.tz_localize(None)

    if mm_type == 'MML':
        color_col      = 'Long Clustering'
        colorbar_title = 'MML Clustering'
    else:
        color_col      = 'Short Clustering'
        colorbar_title = 'MMS Clustering'

    # 2nd-Nearby-Preislogik identisch zum PPCI
    price_col = _ppci_get_price_col(selected_market)
    y_title = 'Price (Continuous Front-Month Proxy) (Report Date)'

    if price_col and not df_futures_prices.empty and price_col in df_futures_prices.columns:
        prices = df_futures_prices[['Date', price_col]].dropna(subset=[price_col]).copy()
        prices = prices.rename(columns={'Date': '_pdate'}).sort_values('_pdate')
        dff = dff.sort_values('_date')
        dff = pd.merge_asof(
            dff, prices,
            left_on='_date', right_on='_pdate',
            direction='backward',
            tolerance=pd.Timedelta(days=7)
        )
        y_vals = pd.to_numeric(dff[price_col], errors='coerce')
    else:
        y_vals = pd.Series([np.nan] * len(dff), index=dff.index)
        y_title = f'Price (keine Daten für {selected_market})'

    # Bubble-Größen (identisch zu PPCI)
    oi = pd.to_numeric(dff['Open Interest'], errors='coerce').fillna(0).abs()
    MIN_PX = 8
    MAX_PX = 30
    sizes_oi = scaled_diameters(oi, min_px=MIN_PX, max_px=MAX_PX)

    color_vals = pd.to_numeric(dff[color_col], errors='coerce')

    # Hover-Text
    dates_str = pd.to_datetime(dff['Date']).dt.strftime('%Y-%m-%d')
    hover_text = [
        f"Date: {d}<br>Price: {p:.2f}<br>Total Open Interest: {o:,.0f}<br>{colorbar_title}: {c:.1f}"
        for d, p, o, c in zip(
            dates_str,
            y_vals.fillna(0),
            oi.fillna(0),
            color_vals.fillna(0)
        )
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dff['Date'],
        y=y_vals,
        mode='markers',
        marker=dict(
            size=sizes_oi,
            sizemode='diameter',
            sizeref=1,
            color=color_vals,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(
                title=colorbar_title,
                thickness=15,
                len=0.75,
                yanchor='middle',
                y=0.5
            ),
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))

    # OI-Bubble-Größen-Legende (identisch zu PPCI)
    base = oi[oi > 0]
    if base.size >= 3:
        q = [0.10, 0.30, 0.50, 0.70, 0.90]
        legend_vals = np.unique(np.round(np.quantile(base, q)).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([50000, 100000, 150000, 200000, 250000], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=MIN_PX, max_px=MAX_PX)

    for v, s in zip(legend_vals, legend_sizes):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=float(s),
                sizemode='diameter',
                sizeref=1,
                color='gray',
                opacity=0.6
            ),
            showlegend=True,
            name=f"{int(v):,}",
            hoverinfo='skip'
        ))

    # Letzter Datenpunkt hervorheben (identisch zu PPCI)
    if not dff.empty and not y_vals.empty:
        fig.add_trace(go.Scatter(
            x=[dff.iloc[-1]['Date']],
            y=[y_vals.iloc[-1]],
            mode='markers',
            marker=dict(
                size=10,
                color='black',
                opacity=1.0,
                line=dict(width=4, color='red')
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f'PP Clustering Indicator ({mm_type}) – {selected_market}',
        xaxis_title='Report Date',
        yaxis_title=y_title,
        legend=dict(
            title=dict(text='Total Open Interest'),
            itemsizing='trace',
            x=1.2,
            y=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=160, t=60, b=60),
        height=600,
    )

    return fig


# ---------------------------------------------------------------------------
# PP Position Size Indicator – Callback
# Wiederverwendet: _ppci_get_price_col, df_futures_prices, merge_asof,
# scaled_diameters, Letzter-Punkt-Highlight aus dem PPCI.
# Bubble-Größe = Anzahl MM-Trader (statt OI), Farbe = Position Size in USD.
# Position Size ($) = MML/MMS Position Size (Kontrakte/Trader) × Price.
# ---------------------------------------------------------------------------
@app.callback(
    Output('pp-position-size-graph', 'figure'),
    [Input('market-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('ppci-position-size-radio', 'value')]
)
def update_pp_position_size(selected_market, start_date, end_date, mm_type):
    dff = df_pivoted[
        (df_pivoted['Market Names'] == selected_market) &
        (df_pivoted['Date'] >= start_date) &
        (df_pivoted['Date'] <= end_date)
    ].copy()

    if dff.empty:
        return go.Figure()

    # Normalize Date to tz-naive for merging (identisch zu PPCI)
    dff['_date'] = pd.to_datetime(dff['Date']).dt.tz_localize(None)

    # 2nd-Nearby-Preislogik identisch zum PPCI
    price_col = _ppci_get_price_col(selected_market)
    y_title = 'Price (Continuous Front-Month Proxy) (Report Date)'

    if price_col and not df_futures_prices.empty and price_col in df_futures_prices.columns:
        prices = df_futures_prices[['Date', price_col]].dropna(subset=[price_col]).copy()
        prices = prices.rename(columns={'Date': '_pdate'}).sort_values('_pdate')
        dff = dff.sort_values('_date')
        dff = pd.merge_asof(
            dff, prices,
            left_on='_date', right_on='_pdate',
            direction='backward',
            tolerance=pd.Timedelta(days=7)
        )
        y_vals = pd.to_numeric(dff[price_col], errors='coerce')
    else:
        y_vals = pd.Series([np.nan] * len(dff), index=dff.index)
        y_title = f'Price (keine Daten für {selected_market})'

    # Position Size in USD = (MML/MMS Position Size in Kontrakten) × Kontraktgröße × Price
    price_series = y_vals.reset_index(drop=True)
    dff = dff.reset_index(drop=True)
    contract_size = _ppci_get_contract_size(selected_market)

    if mm_type == 'MML':
        traders_col    = 'Traders M Money Long'
        pos_size_contr = pd.to_numeric(dff['MML Position Size'], errors='coerce')
        colorbar_title = 'Long Position Size ($)'
        size_legend_title = 'Number of Long Traders'
    else:
        traders_col    = 'Traders M Money Short'
        pos_size_contr = pd.to_numeric(dff['MMS Position Size'], errors='coerce')
        colorbar_title = 'Short Position Size ($)'
        size_legend_title = 'Number of Short Traders'

    # Farbwerte: Position Size in USD
    color_vals = pos_size_contr * contract_size * price_series

    # Bubble-Größe = Anzahl Trader
    traders = pd.to_numeric(dff[traders_col], errors='coerce').fillna(0).abs()
    MIN_PX = 8
    MAX_PX = 30
    sizes_traders = scaled_diameters(traders, min_px=MIN_PX, max_px=MAX_PX)

    # Hover-Text
    dates_str = pd.to_datetime(dff['Date']).dt.strftime('%Y-%m-%d')
    hover_text = [
        f"Date: {d}<br>Price: {p:.2f}<br>{size_legend_title}: {t:,.0f}<br>{colorbar_title}: {c:,.0f}"
        for d, p, t, c in zip(
            dates_str,
            y_vals.fillna(0),
            traders.fillna(0),
            color_vals.fillna(0)
        )
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dff['Date'],
        y=y_vals,
        mode='markers',
        marker=dict(
            size=sizes_traders,
            sizemode='diameter',
            sizeref=1,
            color=color_vals,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(
                title=colorbar_title,
                thickness=15,
                len=0.75,
                yanchor='middle',
                y=0.5
            ),
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))

    # Trader-Anzahl-Legende (Bubble-Größen)
    base = traders[traders > 0]
    if base.size >= 3:
        q = [0.10, 0.30, 0.50, 0.70, 0.90]
        legend_vals = np.unique(np.round(np.quantile(base, q)).astype(int))
        legend_vals = legend_vals[legend_vals > 0]
    else:
        legend_vals = np.array([20, 40, 60, 80, 100], dtype=int)

    legend_sizes = scaled_diameters(legend_vals, min_px=MIN_PX, max_px=MAX_PX)

    for v, s in zip(legend_vals, legend_sizes):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=float(s),
                sizemode='diameter',
                sizeref=1,
                color='gray',
                opacity=0.6
            ),
            showlegend=True,
            name=f"{int(v):,}",
            hoverinfo='skip'
        ))

    # Letzter Datenpunkt hervorheben (identisch zu PPCI)
    if not dff.empty and not y_vals.empty:
        fig.add_trace(go.Scatter(
            x=[dff.iloc[-1]['Date']],
            y=[y_vals.iloc[-1]],
            mode='markers',
            marker=dict(
                size=10,
                color='black',
                opacity=1.0,
                line=dict(width=4, color='red')
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f'PP Position Size Indicator ({mm_type}) – {selected_market}',
        xaxis_title='Report Date',
        yaxis_title=y_title,
        legend=dict(
            title=dict(text=size_legend_title),
            itemsizing='trace',
            x=1.2,
            y=0.5,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=160, t=60, b=60),
        height=600,
    )

    return fig


# ---------------------------------------------------------------------------
# Dry Powder Notional Indicator – Callback
# Orientierung am DP Indicator: Farben, Trendlinien, Most-Recent-Week-Marker,
# Bubble-Sizing und Layout-Parameter werden identisch übernommen.
# Y-Achse: Notional Exposure in USD bn = (MM OI × Kontraktgröße × Price) / 1e9
#   MML (Long):  positiv (oberhalb Nulllinie)
#   MMS (Short): negativ (unterhalb Nulllinie)
# Preisquelle: df_futures_prices via _ppci_get_price_col (identisch zu PPCI).
# ---------------------------------------------------------------------------
@app.callback(
    Output('dp-notional-indicator-graph', 'figure'),
    [Input('market-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_dp_notional(selected_market, start_date, end_date):
    dff = df_pivoted[
        (df_pivoted['Market Names'] == selected_market) &
        (df_pivoted['Date'] >= start_date) &
        (df_pivoted['Date'] <= end_date)
    ].copy().reset_index(drop=True)

    if dff.empty:
        return go.Figure()

    # 2nd-Nearby-Preis (identisch zu PPCI)
    dff['_date'] = pd.to_datetime(dff['Date']).dt.tz_localize(None)
    price_col = _ppci_get_price_col(selected_market)

    if price_col and not df_futures_prices.empty and price_col in df_futures_prices.columns:
        prices = df_futures_prices[['Date', price_col]].dropna(subset=[price_col]).copy()
        prices = prices.rename(columns={'Date': '_pdate'}).sort_values('_pdate')
        dff = dff.sort_values('_date').reset_index(drop=True)
        dff = pd.merge_asof(
            dff, prices,
            left_on='_date', right_on='_pdate',
            direction='backward',
            tolerance=pd.Timedelta(days=7)
        )
        price_series = pd.to_numeric(dff[price_col], errors='coerce')
    else:
        price_series = pd.Series([np.nan] * len(dff), index=dff.index)

    # Notional Exposure in USD bn
    # Formel: Kontrakte × Kontraktgröße (Einheiten/Kontrakt) × Preis (USD/Einheit) / 1e9
    contract_size = _ppci_get_contract_size(selected_market)
    mml_oi = pd.to_numeric(dff['Managed Money Long'],  errors='coerce')
    mms_oi = pd.to_numeric(dff['Managed Money Short'], errors='coerce')

    y_mml =  mml_oi * contract_size * price_series / 1e9   # positiv
    y_mms = -mms_oi * contract_size * price_series / 1e9   # negativ

    x_mml = pd.to_numeric(dff['MML Traders'], errors='coerce')
    x_mms = pd.to_numeric(dff['MMS Traders'], errors='coerce')

    # Bubble-Größe (identisch zu DP Indicator)
    bubble_size = (dff['MML Long OI'].abs() + dff['MML Short OI'].abs())
    desired_max_px = 28
    bsmax = bubble_size.max()
    sizeref = 2.0 * bsmax / (desired_max_px ** 2) if bsmax > 0 else 1.0

    COL_LONG  = "#2c7fb8"  # MML
    COL_SHORT = "#7fcdbb"  # MMS

    fig = go.Figure()

    # Wolken
    fig.add_trace(go.Scatter(
        x=x_mml, y=y_mml,
        mode='markers',
        marker=dict(
            size=bubble_size, sizemode='area', sizeref=sizeref,
            color=COL_LONG, opacity=0.75, line=dict(width=0.6, color='black')
        ),
        name='MML',
        hovertemplate='Traders: %{x}<br>Notional: %{y:.2f} USD bn<extra>MML</extra>'
    ))
    fig.add_trace(go.Scatter(
        x=x_mms, y=y_mms,
        mode='markers',
        marker=dict(
            size=bubble_size, sizemode='area', sizeref=sizeref,
            color=COL_SHORT, opacity=0.75, line=dict(width=0.6, color='black')
        ),
        name='MMS',
        hovertemplate='Traders: %{x}<br>Notional: %{y:.2f} USD bn<extra>MMS</extra>'
    ))

    # Trendlinien (identisch zu DP Indicator: weißer Untergrund + Farblinie)
    x_min = float(min(x_mml.min(), x_mms.min()))
    x_max = float(max(x_mml.max(), x_mms.max()))
    xs = np.array([x_min, x_max])

    def _add_notional_trend(x_s, y_s, color, label):
        mask = x_s.notna() & y_s.notna()
        xv = x_s[mask].astype(float).values
        yv = y_s[mask].astype(float).values
        if len(xv) < 2:
            return
        m, b = np.polyfit(xv, yv, 1)
        ys = m * xs + b
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color='white', width=7),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=3),
            name=label, showlegend=True
        ))

    _add_notional_trend(x_mml, y_mml, COL_LONG,  'MML Trend')
    _add_notional_trend(x_mms, y_mms, COL_SHORT, 'MMS Trend')

    # Most Recent Week (identisch zu DP Indicator)
    fig.add_trace(go.Scatter(
        x=[x_mml.iloc[-1]], y=[y_mml.iloc[-1]],
        mode='markers',
        marker=dict(size=desired_max_px + 4, color='black', line=dict(width=2, color='white')),
        name='Most Recent Week', legendgroup='recent', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[x_mms.iloc[-1]], y=[y_mms.iloc[-1]],
        mode='markers',
        marker=dict(size=desired_max_px + 4, color='black', line=dict(width=2, color='white')),
        name='Most Recent Week', legendgroup='recent', showlegend=False
    ))

    fig.update_layout(
        title='Dry Powder Notional Indicator',
        xaxis=dict(
            title='Number of Traders',
            showgrid=True, gridcolor='LightGray', gridwidth=2, zeroline=False
        ),
        yaxis=dict(
            title='Long and Short $ Exposure (USD bn)',
            showgrid=True, gridcolor='LightGray', gridwidth=2,
            zeroline=True, zerolinecolor='black', zerolinewidth=1
        ),
        plot_bgcolor='white',
        legend_title='Trader Group',
        height=600,
    )

    return fig


# ---------------------------------------------------------------------------
# DP Time Indicator – Callback
# X = Number of Traders (MML / MMS), Y = Long / Short Concentration (%)
# Farbe = Year (wie DP Net Indicators: separate Trace pro Jahr)
# MML: Kreise (oberhalb 0), MMS: Dreiecke (unterhalb 0)
# Konzentration: 100 * MM Long|Short OI / Total OI  (MMS negativ)
# ---------------------------------------------------------------------------
@app.callback(
    Output('dp-time-indicator-graph', 'figure'),
    [Input('market-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_dp_time(selected_market, start_date, end_date):
    dff = df_pivoted[
        (df_pivoted['Market Names'] == selected_market) &
        (df_pivoted['Date'] >= start_date) &
        (df_pivoted['Date'] <= end_date)
    ].copy().reset_index(drop=True)

    if dff.empty:
        return go.Figure()

    # Konzentration (%) – identisch zum PPCI-Callback
    total_oi = pd.to_numeric(dff['Open Interest'], errors='coerce').replace(0, np.nan)
    dff['_y_mml'] =  100.0 * pd.to_numeric(dff['Managed Money Long'],  errors='coerce') / total_oi
    dff['_y_mms'] = -100.0 * pd.to_numeric(dff['Managed Money Short'], errors='coerce') / total_oi

    x_mml = pd.to_numeric(dff['MML Traders'], errors='coerce')
    x_mms = pd.to_numeric(dff['MMS Traders'], errors='coerce')

    fig = go.Figure()

    # Dummy-Traces für Shape-Legende (MML / MMS)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='circle', color='gray', size=9),
        name='MML', legendgroup='shape_mml'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(symbol='triangle-down', color='gray', size=9),
        name='MMS', legendgroup='shape_mms'
    ))

    # Scatter-Traces: pro Jahr eine Farbe, MML=Kreis / MMS=Dreieck
    for year in sorted(dff['Year'].unique()):
        mask = dff['Year'] == year

        fig.add_trace(go.Scatter(
            x=x_mml[mask], y=dff['_y_mml'][mask],
            mode='markers',
            marker=dict(symbol='circle', size=12, opacity=0.75),
            name=str(year), legendgroup=str(year), showlegend=True,
            hovertemplate=(
                f'Year: {year}<br>'
                'Traders: %{x}<br>'
                'Long Conc.: %{y:.1f}%'
                '<extra>MML</extra>'
            )
        ))
        fig.add_trace(go.Scatter(
            x=x_mms[mask], y=dff['_y_mms'][mask],
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, opacity=0.75),
            name=str(year), legendgroup=str(year), showlegend=False,
            hovertemplate=(
                f'Year: {year}<br>'
                'Traders: %{x}<br>'
                'Short Conc.: %{y:.1f}%'
                '<extra>MMS</extra>'
            )
        ))

    # Trendlinien (analog zu DP Notional: weißer Untergrund + Farblinie)
    x_all = pd.concat([x_mml, x_mms]).dropna()
    if not x_all.empty:
        xs = np.array([float(x_all.min()), float(x_all.max())])

        def _add_time_trend(xs_arr, x_s, y_s, color, label):
            mask_t = x_s.notna() & y_s.notna()
            xv = x_s[mask_t].astype(float).values
            yv = y_s[mask_t].astype(float).values
            if len(xv) < 2:
                return
            m, b = np.polyfit(xv, yv, 1)
            ys = m * xs_arr + b
            fig.add_trace(go.Scatter(
                x=xs_arr, y=ys, mode='lines',
                line=dict(color='white', width=7),
                showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=xs_arr, y=ys, mode='lines',
                line=dict(color=color, width=3),
                name=label, showlegend=True
            ))

        _add_time_trend(xs, x_mml, dff['_y_mml'], '#2c7fb8', 'MML Trend')
        _add_time_trend(xs, x_mms, dff['_y_mms'], '#7fcdbb', 'MMS Trend')

    # Most Recent Week (identisch zu DP Notional)
    desired_max_px = 18
    fig.add_trace(go.Scatter(
        x=[x_mml.iloc[-1]], y=[dff['_y_mml'].iloc[-1]],
        mode='markers',
        marker=dict(size=desired_max_px, color='black', line=dict(width=2, color='white')),
        name='Most Recent Week', legendgroup='recent', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[x_mms.iloc[-1]], y=[dff['_y_mms'].iloc[-1]],
        mode='markers',
        marker=dict(size=desired_max_px, color='black', line=dict(width=2, color='white')),
        name='Most Recent Week', legendgroup='recent', showlegend=False
    ))

    fig.update_layout(
        title='Dry Powder Time Indicator',
        xaxis=dict(
            title='Number of Traders',
            showgrid=True, gridcolor='LightGray', gridwidth=2, zeroline=False
        ),
        yaxis=dict(
            title='Long and Short Concentration (%)',
            showgrid=True, gridcolor='LightGray', gridwidth=2,
            zeroline=True, zerolinecolor='black', zerolinewidth=1
        ),
        plot_bgcolor='white',
        legend_title='Year',
        height=600,
    )

    return fig


# ---------------------------------------------------------------------------
# DP Price Indicator – Callback
# X = PMPU Long/Short Traders, Y = PMPU Long/Short OI
# Punktfarbe = Futures-Preis (Continuous Front-Month Proxy, Report Date)
# Trendlinie + Most-Recent-Week-Marker identisch zu DP Notional/Time Indicator
# ---------------------------------------------------------------------------
@app.callback(
    Output('dp-price-indicator-graph', 'figure'),
    [Input('market-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('dp-price-radio', 'value')]
)
def update_dp_price(selected_market, start_date, end_date, pmpu_side):
    dff = df_pivoted[
        (df_pivoted['Market Names'] == selected_market) &
        (df_pivoted['Date'] >= start_date) &
        (df_pivoted['Date'] <= end_date)
    ].copy().reset_index(drop=True)

    if dff.empty:
        return go.Figure()

    if pmpu_side == 'PMPUL':
        x_col    = 'PMPUL Traders'
        y_col    = 'Producer/Merchant/Processor/User Long'
        x_title  = 'PMPU Number of Long Traders'
        y_title  = 'PMPU Long OI (Contracts)'
        pt_title = 'Dry Powder Price Indicator (PMPU Long)'
    else:
        x_col    = 'PMPUS Traders'
        y_col    = 'Producer/Merchant/Processor/User Short'
        x_title  = 'PMPU Number of Short Traders'
        y_title  = 'PMPU Short OI (Contracts)'
        pt_title = 'Dry Powder Price Indicator (PMPU Short)'

    x_vals = pd.to_numeric(dff[x_col], errors='coerce')
    y_vals = pd.to_numeric(dff[y_col], errors='coerce')

    # Futures-Preis als Farbe (identisch zu PPCI: merge_asof, 7-Tage-Toleranz)
    dff['_date'] = pd.to_datetime(dff['Date']).dt.tz_localize(None)
    price_col = _ppci_get_price_col(selected_market)

    if price_col and not df_futures_prices.empty and price_col in df_futures_prices.columns:
        prices = df_futures_prices[['Date', price_col]].dropna(subset=[price_col]).copy()
        prices = prices.rename(columns={'Date': '_pdate'}).sort_values('_pdate')
        dff = dff.sort_values('_date').reset_index(drop=True)
        x_vals = pd.to_numeric(dff[x_col], errors='coerce')
        y_vals = pd.to_numeric(dff[y_col], errors='coerce')
        dff = pd.merge_asof(
            dff, prices,
            left_on='_date', right_on='_pdate',
            direction='backward',
            tolerance=pd.Timedelta(days=7)
        )
        color_vals    = pd.to_numeric(dff[price_col], errors='coerce')
        colorbar_title = 'Price (USD)'
    else:
        color_vals    = pd.Series([np.nan] * len(dff), index=dff.index)
        colorbar_title = 'Price (n/a)'

    # Hover
    dates_str = pd.to_datetime(dff['Date']).dt.strftime('%Y-%m-%d')
    hover_text = [
        f'Date: {d}<br>{x_title}: {x:.0f}<br>{y_title}: {y:,.0f}<br>Price: {c:.2f}'
        for d, x, y, c in zip(
            dates_str,
            x_vals.fillna(0),
            y_vals.fillna(0),
            color_vals.fillna(0)
        )
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers',
        marker=dict(
            size=14,
            color=color_vals,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title=colorbar_title, thickness=15, len=0.75),
            opacity=0.85,
            line=dict(width=0.6, color='black')
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))

    # Most Recent Week (identisch zu DP Notional/Time)
    fig.add_trace(go.Scatter(
        x=[x_vals.iloc[-1]], y=[y_vals.iloc[-1]],
        mode='markers',
        marker=dict(size=18, color='black', line=dict(width=2, color='white')),
        name='Most Recent Week'
    ))

    fig.update_layout(
        title=pt_title,
        xaxis=dict(
            title=x_title,
            showgrid=True, gridcolor='LightGray', gridwidth=2, zeroline=False
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True, gridcolor='LightGray', gridwidth=2, zeroline=False
        ),
        plot_bgcolor='white',
        legend_title='Legend',
        height=600,
    )

    return fig


# ---------------------------------------------------------------------------
# DP Factor (VIX) – Callback
# X = MM Long/Short Traders, Y = MM Long/Short OI
# Punktfarbe = VIX-Wert (FRED via macro_by_date), merge_asof 7-Tage-Toleranz
# Struktur identisch zu DP Price Indicator (ohne Trendlinie)
# ---------------------------------------------------------------------------
@app.callback(
    Output('dp-vix-indicator-graph', 'figure'),
    [Input('market-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('dp-vix-radio', 'value')]
)
def update_dp_vix(selected_market, start_date, end_date, mm_side):
    dff = df_pivoted[
        (df_pivoted['Market Names'] == selected_market) &
        (df_pivoted['Date'] >= start_date) &
        (df_pivoted['Date'] <= end_date)
    ].copy().reset_index(drop=True)

    if dff.empty:
        return go.Figure()

    if mm_side == 'MML':
        x_col   = 'MML Traders'
        y_col   = 'MML Long OI'
        x_title = 'MM Number of Long Traders'
        y_title = 'MM Long OI (Contracts)'
        title   = 'Dry Powder Factor – VIX (MM Long)'
    else:
        x_col   = 'MMS Traders'
        y_col   = 'MMS Short OI'
        x_title = 'MM Number of Short Traders'
        y_title = 'MM Short OI (Contracts)'
        title   = 'Dry Powder Factor – VIX (MM Short)'

    x_vals = pd.to_numeric(dff[x_col], errors='coerce')
    y_vals = pd.to_numeric(dff[y_col], errors='coerce').abs()

    # VIX-Wert als Farbe (FRED via df_macro, merge_asof identisch zu df_futures_prices)
    dff['_date'] = pd.to_datetime(dff['Date']).dt.tz_localize(None)

    if not df_macro.empty and 'vix' in df_macro.columns:
        vix_df = df_macro[['Date', 'vix']].dropna(subset=['vix']).copy()
        vix_df = vix_df.rename(columns={'Date': '_vdate'}).sort_values('_vdate')
        dff = dff.sort_values('_date').reset_index(drop=True)
        x_vals = pd.to_numeric(dff[x_col], errors='coerce')
        y_vals = pd.to_numeric(dff[y_col], errors='coerce').abs()
        dff = pd.merge_asof(
            dff, vix_df,
            left_on='_date', right_on='_vdate',
            direction='backward',
            tolerance=pd.Timedelta(days=7)
        )
        color_vals     = pd.to_numeric(dff['vix'], errors='coerce')
        colorbar_title = 'VIX'
    else:
        color_vals     = pd.Series([np.nan] * len(dff), index=dff.index)
        colorbar_title = 'VIX (n/a)'

    # Hover
    dates_str = pd.to_datetime(dff['Date']).dt.strftime('%Y-%m-%d')
    hover_text = [
        f'Date: {d}<br>{x_title}: {x:.0f}<br>{y_title}: {y:,.0f}<br>VIX: {c:.1f}'
        for d, x, y, c in zip(
            dates_str,
            x_vals.fillna(0),
            y_vals.fillna(0),
            color_vals.fillna(0)
        )
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers',
        marker=dict(
            size=14,
            color=color_vals,
            colorscale='RdYlGn_r',   # hoher VIX = rot (Stress), tiefer VIX = grün
            showscale=True,
            colorbar=dict(title=colorbar_title, thickness=15, len=0.75),
            opacity=0.85,
            line=dict(width=0.6, color='black')
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))

    # Trendlinie (identisch zu DP Notional: weißer Untergrund + Farblinie)
    mask_t = x_vals.notna() & y_vals.notna()
    if mask_t.sum() >= 2:
        xv = x_vals[mask_t].astype(float).values
        yv = y_vals[mask_t].astype(float).values
        xs = np.array([xv.min(), xv.max()])
        m, b = np.polyfit(xv, yv, 1)
        ys = m * xs + b
        col = '#2c7fb8' if mm_side == 'MML' else '#7fcdbb'
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color='white', width=7),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=col, width=3),
            name='Trend', showlegend=True
        ))

    # Most Recent Week (identisch zu DP Price Indicator)
    fig.add_trace(go.Scatter(
        x=[x_vals.iloc[-1]], y=[y_vals.iloc[-1]],
        mode='markers',
        marker=dict(size=18, color='black', line=dict(width=2, color='white')),
        name='Most Recent Week'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=x_title,
            showgrid=True, gridcolor='LightGray', gridwidth=2, zeroline=False
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True, gridcolor='LightGray', gridwidth=2, zeroline=False
        ),
        plot_bgcolor='white',
        legend_title='Legend',
        height=600,
    )

    return fig


# Open browser automatically
if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=8051)
