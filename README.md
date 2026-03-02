# CoT Dashboard - InfluxDB V3

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.18-green.svg)
![InfluxDB](https://img.shields.io/badge/InfluxDB-V3-orange.svg)

Ein interaktives Dashboard zur Visualisierung von **Commitment of Traders (CoT)** Daten, gespeichert in InfluxDB v3 Core, erstellt mit Plotly Dash.

## 🎯 Features

- **Real-time Data Visualization**: Interaktive Dashboards mit Plotly/Dash
- **InfluxDB v3 Integration**: Moderne SQL-basierte Zeitreihendatenbank
- **Multi-Source Data**: Integration von FRED API und Socrata API
- **Responsive Design**: Bootstrap-basiertes responsives Layout
- **Cloud-Ready**: Heroku-Deployment mit Procfile

## 🏗️ Technologie-Stack

- **Backend**: Python 3.12
- **Web Framework**: Dash 2.18, Flask 3.0
- **Database**: InfluxDB v3 Core (SQL)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Dash Bootstrap Components
- **APIs**: FRED API, Socrata API
- **Deployment**: Gunicorn, Heroku

## 📋 Voraussetzungen

- Python 3.12+
- InfluxDB v3 Core Server
- API-Keys (FRED, ggf. andere Datenquellen)

## 🚀 Installation

### 1. Repository klonen

```bash
git clone https://github.com/sinzb1/CoT-Dashboard_InfluxDB-V3.git
cd CoT-Dashboard_InfluxDB-V3
```

### 2. Virtuelle Umgebung erstellen

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows
```

### 3. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 4. InfluxDB v3 konfigurieren

Stelle sicher, dass ein InfluxDB v3 Core Server läuft:

```bash
# Default Configuration:
Host: http://localhost:8181
Database: CoT-Data
Token: [Dein InfluxDB Token]
```

### 5. Umgebungsvariablen setzen

Erstelle eine `.env` Datei im Projekt-Root:

```env
INFLUXDB_HOST=http://localhost:8181
INFLUXDB_TOKEN=your_token_here
INFLUXDB_DATABASE=CoT-Data
FRED_API_KEY=your_fred_api_key
```

## 💻 Verwendung

### Daten laden und schreiben

```bash
python Influx.py
```

### Dashboard starten (lokal)

```bash
python Dash_Lokal.py
```

Das Dashboard ist dann verfügbar unter: **http://127.0.0.1:8051/**

### Produktion (Heroku)

```bash
python app.py
```

## 📊 Projektstruktur

```
CoT-Dashboard_InfluxDB-V3/
├── app.py                  # Haupt-Applikation (Production)
├── Dash_Lokal.py          # Lokale Entwicklungsversion
├── Influx.py              # InfluxDB Daten-Management
├── requirements.txt       # Python Dependencies
├── Procfile               # Heroku Deployment
├── config/
│   └── config.json        # Konfigurationsdateien
└── src/
    ├── clients/           # API-Clients (FRED, Socrata)
    ├── mappings/          # Daten-Mappings
    └── services/          # Business Logic Services
```

## 🔧 Konfiguration

Die Konfiguration erfolgt über:

1. **config/config.json**: Allgemeine App-Konfiguration
2. **.env**: Sensitive Daten (API-Keys, Tokens)
3. **requirements.txt**: Python-Abhängigkeiten

## 🌐 API-Integrationen

### FRED API (Federal Reserve Economic Data)
- Economic indicators
- Rate: Nach API-Key-Typ

### Socrata API
- Open Data Platform
- CoT Report Data

## 🚢 Deployment

### Heroku Deployment

```bash
heroku create your-app-name
git push heroku main
heroku config:set INFLUXDB_TOKEN=your_token
heroku config:set FRED_API_KEY=your_key
```

## 📝 Migration Notes

Dieses Projekt wurde von **InfluxDB v2 (Flux)** auf **InfluxDB v3 Core (SQL)** migriert:

- **Client**: `influxdb-client` → `influxdb3-python`
- **Query Language**: Flux → SQL
- **Data Structure**: Automatisches Pivoting