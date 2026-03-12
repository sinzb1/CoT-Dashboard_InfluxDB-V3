# CoT-Dashboard – Lokale Installationsanleitung

**Projekt:** DIFA – Dashboard for Influx Financial Analysis  
**Technologie:** Python · Dash · InfluxDB v3 Core · Socrata API · FRED API  
**Zielgruppe:** Personen ohne Programmierkenntnisse

---

## Inhaltsverzeichnis

1. [Überblick](#1-überblick)
2. [Voraussetzungen](#2-voraussetzungen)
3. [Installation der benötigten Tools](#3-installation-der-benötigten-tools)
4. [Projekt-Setup](#4-projekt-setup)
5. [InfluxDB v3 Setup](#5-influxdb-v3-setup)
6. [API- und Konfigurationssetup](#6-api--und-konfigurationssetup)
7. [Start des Dashboards](#7-start-des-dashboards)
8. [Funktionstest](#8-funktionstest)
9. [Troubleshooting](#9-troubleshooting)
10. [Checkliste](#10-checkliste)
11. [Offene Punkte / manuell zu prüfen](#11-offene-punkte--manuell-zu-prüfen)

---

## 1. Überblick

Das CoT-Dashboard visualisiert Commitment-of-Traders-Daten (CoT) der CFTC für Edelmetall-Futures (Gold, Silver, Platinum, Palladium, Copper). Es zeigt Positionierungen verschiedener Tradergruppen über verschiedene Indikatoren an.

### Architektur

```
Socrata API (CFTC)  ──┐
                       ├─→  Influx.py  ──→  InfluxDB v3  ──→  Dash_Lokal.py  ──→  Browser
FRED API (Makrodaten) ─┘                     (Datenbank)        (Dashboard)
```

### Ablauf auf einen Blick

| Schritt | Befehl | Zweck |
|---------|--------|-------|
| 1 | InfluxDB v3 starten | Datenbank läuft lokal |
| 2 | `python Influx.py` | Daten von APIs laden & in DB schreiben |
| 3 | `python Dash_Lokal.py` | Dashboard im Browser öffnen |

---

## 2. Voraussetzungen

### 2.1 System

| Anforderung | Minimum | Empfohlen |
|-------------|---------|-----------|
| Betriebssystem | Windows 10 (64-bit) | Windows 11 (64-bit) |
| RAM | 4 GB | 8 GB |
| Festplatte (frei) | 2 GB | 5 GB |
| Internetverbindung | Ja (für API-Abrufe) | Ja |

### 2.2 Software (wird in Schritt 3 installiert)

| Software | Version | Zweck |
|----------|---------|-------|
| Python (Anaconda) | 3.12+ | Ausführung der Python-Skripte |
| InfluxDB v3 Core | 3.8.0+ | Lokale Zeitreihendatenbank |
| VS Code | aktuell | Code-Editor (empfohlen) |

### 2.3 Benötigte API-Zugänge

| API | Zweck | Wo erstellen |
|-----|-------|-------------|
| Socrata App Token | CFTC CoT-Daten laden | [data.sfgov.org/profile/edit/developer_settings](https://data.sfgov.org/profile/edit/developer_settings) |
| FRED API Key | Makrodaten (VIX, USD-Index usw.) | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) |

> **Hinweis:** Beide APIs sind kostenlos nutzbar. Eine Registrierung ist erforderlich.

---

## 3. Installation der benötigten Tools

### 3.1 Python (Anaconda) installieren

1. Öffne folgende Seite im Browser: [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. Lade den **Anaconda-Installer für Windows (64-bit)** herunter
3. Starte den Installer und folge den Anweisungen
4. **Wichtig:** Aktiviere „Add Anaconda to my PATH environment variable"
5. Schliesse den Installer ab

**Verifizierung** (PowerShell öffnen → Rechtsklick auf Start → „Windows PowerShell"):

```powershell
python --version
# Erwartete Ausgabe: Python 3.12.x oder höher
```

### 3.2 InfluxDB v3 Core herunterladen

1. Öffne: [https://github.com/influxdata/influxdb/releases](https://github.com/influxdata/influxdb/releases)
2. Suche nach der neuesten Version (z. B. `3.8.0`)
3. Lade die Datei `influxdb3-core-3.8.0-windows_amd64.zip` herunter
4. Erstelle den Zielordner und entpacke die ZIP-Datei:

```powershell
# Zielordner erstellen
New-Item -ItemType Directory -Path "C:\InfluxDB3" -Force

# ZIP entpacken (Pfad ggf. anpassen)
Expand-Archive -Path "$env:USERPROFILE\Downloads\influxdb3-core-3.8.0-windows_amd64.zip" `
               -DestinationPath "C:\InfluxDB3"
```

5. Überprüfe, ob `influxdb3.exe` vorhanden ist:

```powershell
Get-ChildItem "C:\InfluxDB3\influxdb3-core-3.8.0-windows_amd64"
# Erwartet: influxdb3.exe, LICENSE, README.md
```

### 3.3 VS Code installieren (optional, empfohlen)

1. Öffne: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
2. Lade den Windows-Installer herunter und führe ihn aus
3. Empfohlene Extension: **Python** (von Microsoft)

---

## 4. Projekt-Setup

### 4.1 Projektdateien beschaffen

Das Projekt befindet sich im Ordner `C:\DIFA_influxv3`. Falls noch nicht vorhanden, kopiere den gesamten Projektordner dorthin.

```powershell
# In den Projektordner wechseln
cd C:\DIFA_influxv3
```

Die Projektstruktur sieht so aus:

```
DIFA_influxv3/
├── Influx.py              ← Daten laden & in InfluxDB schreiben
├── Dash_Lokal.py          ← Dashboard (lokaler Start)
├── app.py                 ← Einfache Test-App
├── requirements.txt       ← Python-Abhängigkeiten
├── config/
│   └── config.json        ← API-Schlüssel (muss ausgefüllt werden!)
└── src/
    ├── clients/           ← API-Client-Klassen
    └── services/          ← Datenlogik
```

### 4.2 Python-Pakete installieren

```powershell
cd C:\DIFA_influxv3
pip install -r requirements.txt
```

> Die Installation kann 2–5 Minuten dauern. Fehlermeldungen zu bereits installierten Paketen können ignoriert werden.

**Zusätzliche Pakete, die möglicherweise nicht in der requirements.txt enthalten sind:**

```powershell
pip install fredapi sodapy
```

**Verifizierung:**

```powershell
python -c "import influxdb_client_3; print('OK:', influxdb_client_3.__version__)"
# Erwartete Ausgabe: OK: 0.18.0
```

---

## 5. InfluxDB v3 Setup

> ### Überblick: Welche Fenster werden benötigt?
>
> Für den Betrieb des Dashboards müssen **gleichzeitig drei PowerShell-Fenster** offen sein. Öffne sie **vor** Schritt 5.2 und benenne sie gedanklich so:
>
> | Fenster | Zweck | Darf geschlossen werden? |
> |---------|-------|--------------------------|
> | **Fenster A – InfluxDB-Server** | Datenbank läuft hier dauerhaft | ❌ Nein – niemals während Betrieb |
> | **Fenster B – Token / Admin** | Einmalig für Token-Erstellung | ✅ Ja – nach Schritt 5.3 |
> | **Fenster C – Projektbefehle** | `Influx.py` und `Dash_Lokal.py` starten | ✅ Ja – nach Bedarf |
>
> **PowerShell öffnen:** Rechtsklick auf das Windows-Start-Symbol → „Windows PowerShell" oder „Terminal"

---

### 5.1 Datenverzeichnis anlegen

**Fenster: Fenster B (oder ein beliebiges PowerShell-Fenster)**

Dieser Schritt legt den Ordner an, in dem InfluxDB alle Daten dauerhaft speichert.

```powershell
New-Item -ItemType Directory -Path "$env:USERPROFILE\.influxdb" -Force
```

**Erwartete Ausgabe:**
```
    Verzeichnis: C:\Users\<Dein-Benutzername>

Mode                 LastWriteTime         Length Name
----                 -------------         ------  ----
d----         08.03.2026    10:00                .influxdb
```

Falls der Ordner bereits existiert, erscheint keine Ausgabe — das ist normal und kein Fehler.

**Wo werden die Daten gespeichert?**  
Unter `C:\Users\<Benutzername>\.influxdb` — also im persönlichen Windows-Profilordner. Die Daten bleiben nach einem PC-Neustart erhalten.

---

### 5.2 InfluxDB v3 Server starten (Fenster A – dauerhaft offen lassen!)

> ⚠️ **Wichtig:** Dieser Befehl muss in einem **eigenen, dauerhaft offenen PowerShell-Fenster** ausgeführt werden.  
> **Schliesse dieses Fenster nicht**, solange du das Dashboard nutzt — der Server würde sonst sofort stoppen.

**Vorgehen Schritt für Schritt:**

**1.** Öffne ein **neues** PowerShell-Fenster (das wird Fenster A).  
&nbsp;&nbsp;&nbsp;&nbsp;→ Rechtsklick auf Start → „Windows PowerShell"

**2.** Wechsle in den InfluxDB-Ordner:

```powershell
cd "C:\InfluxDB3\influxdb3-core-3.8.0-windows_amd64"
```

> Falls du InfluxDB in einem anderen Verzeichnis entpackt hast, passe den Pfad entsprechend an.  
> Prüfe den genauen Ordnernamen mit: `Get-ChildItem "C:\InfluxDB3"`

**3.** Starte den Server:

```powershell
.\influxdb3.exe serve `
    --object-store file `
    --data-dir "$env:USERPROFILE\.influxdb" `
    --node-id "influxdb-difa-node"
```

**Erwartete Ausgabe (Fenster A, nach wenigen Sekunden):**
```
INFO influxdb3_server: startup time: 263ms address=0.0.0.0:8181
INFO influxdb3_server: InfluxDB 3 Core started successfully
```

**Was bedeuten die Parameter?**

| Parameter | Bedeutung |
|-----------|-----------|
| `--object-store file` | Daten werden als Dateien gespeichert (kein Cloud-Speicher) |
| `--data-dir` | Pfad zum Datenordner (aus Schritt 5.1) |
| `--node-id` | Interner Name dieser Datenbankinstanz (beliebig, aber konsistent halten) |

**Der Server läuft nun — Fenster A bleibt offen!**  
Du siehst laufend Log-Meldungen. Das ist normal. Fehler erscheinen in roter Schrift.

---

### 5.2.1 Server-Verfügbarkeit prüfen

**Fenster: Fenster B (neues PowerShell-Fenster öffnen!)**

Öffne ein weiteres PowerShell-Fenster (Fenster A bleibt offen!) und führe aus:

```powershell
curl http://localhost:8181/health
```

**Mögliche Antworten und ihre Bedeutung:**

| Antwort | Bedeutung |
|---------|-----------|
| `{"error":"the request was not authenticated"}` | ✅ Server läuft korrekt, Token fehlt noch |
| `{"status":"ok"}` | ✅ Server läuft und ist bereit |
| `curl: connection refused` | ❌ Server läuft nicht — zurück zu Schritt 5.2 |
| Kein Output / Timeout | ❌ Falscher Port oder Firewall blockiert — Port 8181 prüfen |

> Die Meldung `not authenticated` ist **kein Fehler** — sie bedeutet, dass der Server erreichbar ist und auf einen Token wartet.

---

### 5.3 Admin-Token erstellen (Fenster B – einmalig)

> ⚠️ **Achtung:** 
> - Dieser Schritt wird nur **einmalig** beim ersten Setup ausgeführt
> - Der Token wird nur **einmal** in der Konsole angezeigt — kopiere ihn sofort!
> - Führe diesen Befehl in **Fenster B** aus (nicht in Fenster A, wo der Server läuft)

**Vorgehen:**

**1.** In **Fenster B** (nicht in Fenster A!) in den InfluxDB-Ordner wechseln:

```powershell
cd "C:\InfluxDB3\influxdb3-core-3.8.0-windows_amd64"
```

**2.** Token erstellen:

```powershell
.\influxdb3.exe create token --admin
```

**Erwartete Ausgabe:**
```
New token created successfully!
Token: apiv3_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

IMPORTANT: Store this token securely, as it will not be shown again.
```

**3.** Token sofort kopieren und sicher speichern!

```powershell
# Token in eine Textdatei ausserhalb des Projektordners schreiben
# (Ersetze "apiv3_DEIN_TOKEN_HIER" durch deinen echten Token)
Set-Content -Path "C:\InfluxDB3\admin_token.txt" -Value "apiv3_DEIN_TOKEN_HIER"
```

> **Warum ausserhalb des Projektordners?** Der Projektordner wird möglicherweise in Git eingecheckt oder geteilt. Tokens dürfen nie in Git-Repositories gespeichert werden.

**Was tun, wenn der Token verloren geht?**  
Einen neuen Token erstellen: `.\influxdb3.exe create token --admin` — alten Dateien ebenfalls aktualisieren.

---

### 5.4 Token in den Python-Dateien eintragen (Fenster C oder VS Code)

**Fenster: VS Code Editor oder ein beliebiges PowerShell-Fenster (Fenster C)**

Öffne die folgenden drei Dateien im Editor (z. B. VS Code) und ersetze den Platzhalter `YOUR_INFLUXDB_TOKEN_HERE` durch deinen echten Token aus Schritt 5.3.

> **Tipp für VS Code:** Nutze „Suchen und Ersetzen" (Strg+H) zum schnellen Ersetzen in jeder Datei.

**Datei 1: `Influx.py`** (Zeile ~10, direkt nach den Imports):
```python
token = "YOUR_INFLUXDB_TOKEN_HERE"   # ← Hier deinen Token eintragen
database = "CoT-Data"
host = "http://localhost:8181"
```

**Datei 2: `Dash_Lokal.py`** (Zeile ~18, im Verbindungsabschnitt):
```python
host = "http://localhost:8181"
token = "YOUR_INFLUXDB_TOKEN_HERE"   # ← Hier deinen Token eintragen
database = "CoT-Data"
```

**Datei 3: `app.py`** (Zeile ~13):
```python
token = "YOUR_INFLUXDB_TOKEN_HERE"   # ← Hier deinen Token eintragen
host = "http://localhost:8181"
database = "CoT-Data"
```

**Dateien speichern:** Strg+S in jeder Datei nach der Änderung.

---

### 5.5 Zusammenfassung: Fensterstatus nach Abschluss von Kapitel 5

| Fenster | Status | Inhalt |
|---------|--------|--------|
| **Fenster A – InfluxDB-Server** | 🟢 offen & aktiv | Server-Logs laufen durch |
| **Fenster B – Token / Admin** | 🟡 kann offen bleiben | Token-Ausgabe sichtbar |
| **Fenster C – Projektbefehle** | ⬜ noch nicht verwendet | Wird in Kapitel 7 benötigt |

---

## 6. API- und Konfigurationssetup

### 6.1 Übersicht der benötigten Schlüssel

| Variable | Datei | Pfad im JSON | Beschreibung |
|----------|-------|-------------|--------------|
| Socrata App Token | `config/config.json` | `socrata.app_token` | Zugriff auf CFTC-Daten |
| FRED API Key | `config/config.json` | `fred.app_token` | Zugriff auf Makrodaten (VIX, USD) |
| InfluxDB Token | `Influx.py`, `Dash_Lokal.py`, `app.py` | direkt im Code | Authentifizierung InfluxDB v3 |

### 6.2 config.json befüllen

Öffne die Datei `config/config.json` und ersetze die Platzhalter:

```json
{
  "socrata": {
    "domain": "publicreporting.cftc.gov",
    "app_token": "YOUR_SOCRATA_APP_TOKEN_HERE",
    "limit": 5000
  },
  "fred": {
    "app_token": "YOUR_FRED_API_KEY_HERE"
  }
}
```

#### Socrata App Token erstellen

1. Gehe zu: [https://publicreporting.cftc.gov/profile/edit/developer_settings](https://publicreporting.cftc.gov/profile/edit/developer_settings)
2. Registriere dich oder melde dich an
3. Klicke auf „Create New App Token"
4. Kopiere den Token und füge ihn unter `socrata.app_token` ein

#### FRED API Key erstellen

1. Gehe zu: [https://fredaccount.stlouisfed.org/apikeys](https://fredaccount.stlouisfed.org/apikeys)
2. Registriere dich (kostenlos) oder melde dich an
3. Klicke auf „Request API Key"
4. Kopiere den Key und füge ihn unter `fred.app_token` ein

> **Sicherheitshinweis:** Schreibe niemals echte API-Schlüssel in öffentliche Repositories (GitHub). Die `config.json` sollte in der `.gitignore` stehen.

---

## 7. Start des Dashboards

### 7.1 Voraussetzungen vor dem Start

Stelle sicher, dass:
- [ ] InfluxDB v3 Server läuft (Schritt 5.2)
- [ ] `config/config.json` mit echten API-Keys befüllt ist
- [ ] InfluxDB-Token in allen Python-Dateien eingetragen ist

### 7.2 Schritt 1: Daten in InfluxDB schreiben

Öffne PowerShell im Projektordner:

```powershell
cd C:\DIFA_influxv3
python Influx.py
```

**Erwartete Ausgabe:**
```
[SocrataClient] Total rows to fetch: 2299 for ...
[SocrataClient] Retrieved 2299 rows for ...
Writing 2299 CoT data points to InfluxDB v3...
CoT data write completed.
Writing XXX macro data points to InfluxDB v3...
Successfully wrote XXX macro data points.
InfluxDB v3 client closed. Migration write complete!
```

> Dieser Schritt kann 5–15 Minuten dauern (abhängig von der Internetverbindung). Er muss nur beim ersten Start oder für Updates ausgeführt werden.

### 7.3 Schritt 2: Dashboard starten

```powershell
cd C:\DIFA_influxv3
python Dash_Lokal.py
```

**Erwartete Ausgabe:**
```
Connecting to InfluxDB v3...
Fetching data from InfluxDB v3...
Fetched XXXX rows from InfluxDB v3
Dash is running on http://127.0.0.1:8051/
```

Das Dashboard öffnet sich automatisch im Browser unter: **http://127.0.0.1:8051/**

---

## 8. Funktionstest

Prüfe nach dem Start folgende Punkte im Browser:

| Test | Erwartetes Ergebnis |
|------|---------------------|
| Seite lädt | Dashboard mit Navbar „COT-Data Overview/Analysis Dashboard" sichtbar |
| Market-Dropdown | Werte auswählbar: Gold, Silver, Platinum, Palladium, Copper |
| Datumswähler | Start- und Enddatum lässt sich einstellen |
| Übersichtstabelle | Zeigt Trader-Gruppen mit Positions-Balken |
| Clustering-Indikator | Graph lädt mit Datenpunkten |
| Position Size Indicator | Mindestens 4 Graphen sichtbar (PMPU, SD, MM, OR) |
| Dry Powder Indicator | Bubble-Chart lädt |

**Datenbank-Inhalt prüfen** (optional, in PowerShell):

```powershell
$headers = @{Authorization = "Bearer apiv3_DEIN_TOKEN_HIER"}
Invoke-WebRequest -Uri "http://localhost:8181/health" -Headers $headers -UseBasicParsing
# Erwartete Ausgabe: StatusCode 200
```

---

## 9. Troubleshooting

### 9.1 Häufige Fehler und Lösungen

| Fehlerbild | Mögliche Ursache | Lösung |
|------------|-----------------|--------|
| `ModuleNotFoundError: No module named 'dash'` | Falsche Python-Version aktiv (z. B. Python 3.13 statt Anaconda) | Anaconda Python explizit aufrufen: `C:\Users\<Name>\anaconda3\python.exe Dash_Lokal.py` |
| `ModuleNotFoundError: No module named 'influxdb_client_3'` | Paket nicht installiert | `pip install influxdb3-python` |
| `ModuleNotFoundError: No module named 'fredapi'` | Paket nicht installiert | `pip install fredapi sodapy` |
| `401 Unauthorized` beim Schreiben/Lesen | Falscher oder abgelaufener InfluxDB-Token | Neuen Token erstellen (Schritt 5.3) und in alle Python-Dateien eintragen |
| `Connection refused` / `ConnectionError` | InfluxDB-Server läuft nicht | Server starten (Schritt 5.2); Port 8181 prüfen |
| `AttributeError: 'pyarrow.lib.Table' has no attribute 'rename'` | Query gibt PyArrow-Objekt zurück statt DataFrame | In `Dash_Lokal.py` sicherstellen: `df_pivoted = table.to_pandas()` |
| `Got an unexpected keyword argument 'write_options'` | Falsche API-Aufruf-Signatur für v3 | `client.write(record=point)` ohne `write_options`-Parameter verwenden |
| Dashboard lädt, aber Graphen leer | Keine Daten in InfluxDB | `python Influx.py` ausführen, um Daten zu laden |
| `KeyError: 'Total Traders'` | Spaltennamen nach Migration abweichend | Abfrage prüfen; InfluxDB v3 gibt Daten bereits pivotiert zurück |
| Port 8181 bereits belegt | Anderer Prozess nutzt Port 8181 | `netstat -ano \| findstr :8181` – den Prozess beenden oder anderen Port im Server-Startbefehl wählen |

### 9.2 Richtige Python-Version sicherstellen

```powershell
# Prüfen, welches Python aktiv ist
where.exe python

# Falls Anaconda nicht an erster Stelle steht, direkt mit Anaconda starten:
C:\Users\<Benutzername>\anaconda3\python.exe Dash_Lokal.py
```

### 9.3 InfluxDB-Verbindung debuggen

```powershell
# 1. Server-Status prüfen (ohne Token)
curl http://localhost:8181/health
# Zeigt "the request was not authenticated" → Server läuft korrekt

# 2. Mit Token prüfen
$headers = @{Authorization = "Bearer apiv3_DEIN_TOKEN"}
Invoke-WebRequest -Uri "http://localhost:8181/health" -Headers $headers -UseBasicParsing
# StatusCode 200 → Verbindung und Token korrekt
```

### 9.4 Daten in InfluxDB prüfen

Füge diesen Debug-Code temporär in eine Python-Datei ein:

```python
from influxdb_client_3 import InfluxDBClient3

client = InfluxDBClient3(
    host="http://localhost:8181",
    token="apiv3_DEIN_TOKEN",
    database="CoT-Data"
)
result = client.query("SELECT COUNT(*) FROM cot_data", language="sql")
print(result)
client.close()
```

### 9.5 Logs des InfluxDB-Servers einsehen

Im PowerShell-Fenster, in dem der Server läuft, werden alle Aktivitäten ausgegeben. Fehlermeldungen erscheinen dort in Echtzeit.

---

## 10. Checkliste

Nutze diese Checkliste, um eine erfolgreiche Installation zu bestätigen:

### Installation

- [ ] Python (Anaconda) ist installiert (`python --version` zeigt 3.12+)
- [ ] `pip install -r requirements.txt` wurde ohne kritische Fehler abgeschlossen
- [ ] `pip install fredapi sodapy` wurde ausgeführt
- [ ] `python -c "import influxdb_client_3"` gibt keinen Fehler

### InfluxDB v3

- [ ] `influxdb3.exe` liegt unter `C:\InfluxDB3\...`
- [ ] Server startet fehlerfrei und lauscht auf Port 8181
- [ ] Admin-Token wurde erstellt und sicher gespeichert
- [ ] Token wurde in `Influx.py`, `Dash_Lokal.py` und `app.py` eingetragen
- [ ] `curl http://localhost:8181/health` antwortet (auch „not authenticated" ist ein Zeichen, dass der Server läuft)

### API-Konfiguration

- [ ] `config/config.json` enthält gültige Socrata App Token
- [ ] `config/config.json` enthält gültigen FRED API Key
- [ ] Socrata-Abfrage funktioniert (`Influx.py` zeigt „Retrieved X rows")
- [ ] FRED-Abfrage funktioniert (`Influx.py` schreibt Makrodaten)

### Dashboard

- [ ] `python Influx.py` läuft fehlerfrei durch
- [ ] `python Dash_Lokal.py` startet ohne Fehler
- [ ] Browser öffnet http://127.0.0.1:8051/ automatisch
- [ ] Market-Dropdown zeigt Gold, Silver, Platinum, Palladium, Copper
- [ ] Mindestens ein Graph lädt korrekt mit Daten

---

## 11. Offene Punkte / manuell zu prüfen

Die folgenden Punkte konnten nicht vollständig aus dem Projektcode abgeleitet werden und müssen individuell geprüft werden:

### 11.1 Sicherheit

> **Kritisch:** API-Schlüssel und InfluxDB-Token sind aktuell **direkt im Code** hinterlegt (`config/config.json`, `Influx.py`, `Dash_Lokal.py`, `app.py`). Für sicheres Setup sollten diese in Umgebungsvariablen oder einer `.env`-Datei gespeichert werden.

Empfohlenes Vorgehen:
1. Datei `.env` erstellen und in `.gitignore` aufnehmen
2. Pakete `python-dotenv` installieren
3. Werte mit `os.getenv("INFLUXDB_TOKEN")` auslesen

### 11.2 Produktionsreife

- Das Dashboard ist für den **lokalen Betrieb** ausgelegt (`Dash_Lokal.py`, Port 8051)
- Für einen öffentlichen Deployment-Betrieb (z. B. via `app.py` mit `gunicorn`) sind weitere Konfigurationsschritte nötig, die im Code nicht vollständig dokumentiert sind
- `gunicorn` ist in der `requirements.txt` enthalten, aber kein Startbefehl ist definiert

### 11.3 Datenpersistenz bei Server-Neustart

- Nach einem Neustart des Computers muss der InfluxDB v3 Server **manuell neu gestartet** werden (Schritt 5.2)
- Die Daten bleiben erhalten (unter `C:\Users\<Name>\.influxdb`)
- `Influx.py` muss **nicht** erneut ausgeführt werden, solange die Daten in der DB vorhanden sind

### 11.4 Datenaktualität

- `Influx.py` muss **manuell** erneut ausgeführt werden, um neue CoT-Daten zu laden
- Ein automatischer Update-Mechanismus (Scheduler, Cron) ist im Projekt nicht implementiert
- Die CFTC veröffentlicht neue CoT-Daten wöchentlich (freitags)

### 11.5 FRED-Datenreihen

- Es ist nicht vollständig dokumentiert, welche FRED-Datenreihen abgerufen werden
- Im Code sichtbar: VIX (`VIXCLS`), USD-Index (`DTWEXBGS`) sowie weitere Reihen in `fred_api_data_service.py`
- Bei API-Fehlern zu FRED-Daten prüfen, ob der API-Key aktiv ist und die Reihen noch existieren

### 11.6 Token-Ablauf

- InfluxDB v3 Core Admin-Tokens laufen aktuell **nicht ab**, können aber manuell widerrufen werden
- Bei `401 Unauthorized`-Fehlern nach einem Server-Neustart muss ein **neuer Token** erstellt werden, da beim Neustart ein neuer Datenkontext entsteht (je nach `--data-dir`-Konfiguration)
