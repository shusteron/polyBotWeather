#!/bin/bash
# Elite Weather Bot — wrapper script
# Called by crontab for all scheduled tasks

BOT_DIR="/Users/ronshuster/poly bot"
PYTHON="/usr/bin/python3"
LOG="$BOT_DIR/data/cron.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

cd "$BOT_DIR" || exit 1

MODE="${1:-scan}"

echo "" >> "$LOG"
echo "========================================" >> "$LOG"
echo "[$TIMESTAMP] Starting mode: $MODE" >> "$LOG"

# Run the bot
$PYTHON main.py --mode "$MODE" >> "$LOG" 2>&1
EXIT_CODE=$?

echo "[$TIMESTAMP] Finished mode: $MODE (exit=$EXIT_CODE)" >> "$LOG"

# After scan — also export Excel report
if [ "$MODE" = "scan" ]; then
    echo "[$TIMESTAMP] Exporting Excel report..." >> "$LOG"
    $PYTHON main.py --mode report >> "$LOG" 2>&1

    # Push updated data and report to GitHub
    cd "$BOT_DIR"
    git add data/ exports/ >> "$LOG" 2>&1
    git commit -m "auto: scan $(date '+%Y-%m-%d %H:%M')" >> "$LOG" 2>&1
    git push origin main >> "$LOG" 2>&1
    echo "[$TIMESTAMP] GitHub push done" >> "$LOG"
fi

# After resolve — push calibration data
if [ "$MODE" = "resolve" ]; then
    cd "$BOT_DIR"
    git add data/ >> "$LOG" 2>&1
    git commit -m "auto: resolve $(date '+%Y-%m-%d %H:%M')" >> "$LOG" 2>&1
    git push origin main >> "$LOG" 2>&1
fi
