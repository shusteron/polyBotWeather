# Elite Weather Prediction Bot — Command Reference

## First-Time Setup (run once)

```bash
cd ~/poly\ bot
pip3 install -r requirements.txt
```

---

## Daily Commands

### Scan for market opportunities
```bash
cd ~/poly\ bot && python3 main.py --mode scan
```
**When to run:** Every morning and evening during active periods.
**How long it takes:** 30–90 seconds.
**What it does:** Fetches all active Polymarket weather markets, pulls ensemble forecasts,
calculates edge, and logs any opportunities found. If nothing meets the confidence threshold,
it logs NO TRADE (this is correct behavior).

---

### Check portfolio status
```bash
cd ~/poly\ bot && python3 main.py --mode status
```
**When to run:** Anytime — instant.
**What it shows:** Paper capital, open positions, win rate, total PnL.

---

### Close expired positions and record outcomes
```bash
cd ~/poly\ bot && python3 main.py --mode resolve
```
**When to run:** Once a day, in the morning.
**What it does:** Checks markets that have passed their end date, marks positions
as resolved, and updates calibration data.

---

### Export full report to Excel
```bash
cd ~/poly\ bot && python3 main.py --mode report
```
**When to run:** Weekly, or whenever you want to review performance.
**Output:** Creates a dated .xlsx file in the `exports/` folder with 8 sheets:
Trades, Skipped Trades, Forecast Stability, Ensemble Analysis,
Provider Accuracy, Calibration Metrics, Drawdown Analysis, Performance Metrics.

---

### Run continuously (auto-scan loop every hour)
```bash
cd ~/poly\ bot && python3 main.py --mode run
```
**When to run:** If you want fully automated paper trading.
**How long to keep running:** See schedule below.
**To stop:** Press Ctrl+C

---

## How Long to Keep the Bot Running

| Period | Recommended schedule | Why |
|--------|---------------------|-----|
| **Now – May 31** | Scan 2x/day (morning + evening) | Hurricane season pre-markets closing |
| **June 1 – November 30** | Scan every 1–4 hours | Atlantic hurricane season — most weather markets |
| **July – August** | Scan every 1–2 hours | Daily temperature markets peak activity |
| **December – May** | Scan 1x/day | Slower market period |

**The bot does NOT need to run 24/7.** Weather markets move slowly.
Running it 2–4 times per day is sufficient. The edge comes from analysis quality,
not from being first.

---

## Run on a Schedule (Mac, runs automatically)

To scan every 4 hours automatically without keeping a terminal open:

```bash
# Add to crontab — runs scan at 07:00, 11:00, 15:00, 19:00 daily
crontab -e
```

Add this line (paste it in the editor that opens):
```
0 7,11,15,19 * * * cd /Users/ronshuster/poly\ bot && python3 main.py --mode scan >> /Users/ronshuster/poly\ bot/data/cron.log 2>&1
```

Save and exit (press Escape, then type `:wq`, then Enter).

---

## What the Bot Is Looking For

The bot only signals a trade when ALL of these are true:

1. **Weather market exists** on Polymarket (temperature, precipitation, hurricane)
2. **Edge > threshold** — model probability is meaningfully higher than market price
3. **Ensemble spread is low** — models agree (low uncertainty)
4. **Forecast stable** across multiple runs — no volatile flip-flopping
5. **Threshold distance is large** — forecast is far from the yes/no boundary
6. **Liquidity sufficient** — real money in the market
7. **Confidence score ≥ 65/100**

If ANY condition fails → NO TRADE (this is the intended behavior).

---

## API Status

| Source | Status | Notes |
|--------|--------|-------|
| Polymarket weather tab | ✅ Live | 171+ active markets |
| GFS ensemble (Open-Meteo) | ✅ Live | 31 ensemble members |
| ECMWF (Open-Meteo) | ⚠️ Deterministic only | Free tier limitation |
| NOAA NWS | ✅ Live | US cities only |
| METAR observations | ✅ Live | Airport weather stations |
| ERA5 historical | ❌ Not connected | Requires CDS account |
| Meteostat historical | ❌ Not connected | To be added |

---

## Log Files

```bash
cat ~/poly\ bot/data/bot.log          # Full activity log
ls ~/poly\ bot/exports/               # Excel report files
cat ~/poly\ bot/data/trades.json      # All paper trades
cat ~/poly\ bot/data/calibration.json # Calibration history
```
