#!/usr/bin/env python3
"""
Elite Weather Prediction Market Bot
Paper trading mode — no real funds at risk.

Usage:
    python main.py --mode scan       # Run a market scan cycle
    python main.py --mode report     # Export Excel/CSV report
    python main.py --mode resolve    # Try to resolve expired market positions
    python main.py --mode status     # Print portfolio status
"""
import argparse
import sys
import os

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from src.bot import EliteWeatherBot


def main():
    parser = argparse.ArgumentParser(
        description="Elite Weather Prediction Market Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  scan     Scan Polymarket for weather markets and evaluate trading signals
  report   Generate a full Excel performance report
  resolve  Attempt to resolve expired open positions via NOAA data
  status   Print current portfolio status

Examples:
  python main.py --mode scan
  python main.py --mode report --output exports/
  python main.py --mode status
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["scan", "report", "resolve", "status"],
        default="scan",
        help="Operating mode (default: scan)",
    )
    parser.add_argument(
        "--output",
        default="exports/",
        help="Output directory for reports (default: exports/)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for persistent data (default: data)",
    )

    args = parser.parse_args()

    bot = EliteWeatherBot(config_path=args.config, data_dir=args.data_dir)

    if args.mode == "scan":
        print("Starting market scan cycle...")
        bot.run_scan_cycle()
        print("Scan complete. Run with --mode status to see portfolio.")

    elif args.mode == "report":
        print(f"Generating report in {args.output} ...")
        path = bot.export_report(args.output)
        print(f"Report exported to {path}")

    elif args.mode == "resolve":
        print("Attempting to resolve expired positions...")
        bot.resolve_expired_markets()
        print("Resolution pass complete.")

    elif args.mode == "status":
        bot.print_status()


if __name__ == "__main__":
    main()
