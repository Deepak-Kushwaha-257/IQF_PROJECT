# """
# scripts/download_fred_data.py
# ==============================
# Downloads US Treasury Par Yield Curve data from FRED.

# Prerequisites:
#     pip install fredapi python-dotenv
#     Set FRED_API_KEY in .env file or environment variable.

# Usage:
#     python scripts/download_fred_data.py
#     python scripts/download_fred_data.py --start 2000-01-01 --end 2023-02-16
# """

# import os
# import argparse
# import pandas as pd

# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# FRED_TICKERS = {
#     "DGS3MO": "3m",
#     "DGS6MO": "6m",
#     "DGS1":   "1y",
#     "DGS2":   "2y",
#     "DGS3":   "3y",
#     "DGS5":   "5y",
#     "DGS10":  "10y",
#     "DGS20":  "20y",
#     "DGS30":  "30y",
# }


# def download_fred(api_key: str, start: str, end: str, out_dir: str):
#     from fredapi import Fred
#     fred = Fred(api_key=api_key)
#     os.makedirs(out_dir, exist_ok=True)

#     print(f"Downloading {len(FRED_TICKERS)} series from FRED ({start} → {end})...")

#     all_series = {}
#     for ticker, name in FRED_TICKERS.items():
#         print(f"  {ticker} ({name})...", end=" ")
#         try:
#             s = fred.get_series(ticker, observation_start=start, observation_end=end)
#             s.name = name
#             # Save individual CSV
#             s.to_csv(os.path.join(out_dir, f"{ticker}.csv"), header=True)
#             all_series[name] = s
#             print(f"OK ({len(s)} obs)")
#         except Exception as e:
#             print(f"FAILED: {e}")

#     # Save combined CSV
#     combined = pd.DataFrame(all_series)
#     combined.index.name = "date"
#     combined.to_csv(os.path.join(out_dir, "usdyc_combined.csv"))
#     print(f"\nSaved combined file: {out_dir}/usdyc_combined.csv")
#     print(f"Shape: {combined.shape}")
#     print(f"Date range: {combined.index[0]} → {combined.index[-1]}")
#     print("\nFirst 5 rows:")
#     print(combined.head())
#     print("\nMissing values per tenor:")
#     print(combined.isnull().sum())


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--start",   default="2000-01-01")
#     parser.add_argument("--end",     default="2023-02-16")
#     parser.add_argument("--out_dir", default="data/raw")
#     args = parser.parse_args()

#     api_key = os.environ.get("FRED_API_KEY")
#     if not api_key:
#         print("ERROR: FRED_API_KEY not set.")
#         print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
#         print("Then create a .env file with:  FRED_API_KEY=your_key_here")
#         exit(1)

#     download_fred(api_key, args.start, args.end, args.out_dir)

"""
scripts/download_fred_data.py
==============================
Downloads US Treasury Par Yield Curve data from FRED.

Prerequisites:
    pip install fredapi python-dotenv
    Set FRED_API_KEY in .env file or environment variable.

Usage:
    python scripts/download_fred_data.py
    python scripts/download_fred_data.py --start 2000-01-01 --end 2023-02-16
"""

import os
import argparse
import pandas as pd

# ✅ FIX: Load .env from project root (guaranteed working)
try:
    from dotenv import load_dotenv

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENV_PATH = os.path.join(BASE_DIR, ".env")

    load_dotenv(dotenv_path=ENV_PATH)

except ImportError:
    print("python-dotenv not installed. Skipping .env loading.")


FRED_TICKERS = {
    "DGS3MO": "3m",
    "DGS6MO": "6m",
    "DGS1":   "1y",
    "DGS2":   "2y",
    "DGS3":   "3y",
    "DGS5":   "5y",
    "DGS10":  "10y",
    "DGS20":  "20y",
    "DGS30":  "30y",
}


def download_fred(api_key: str, start: str, end: str, out_dir: str):
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {len(FRED_TICKERS)} series from FRED ({start} → {end})...")

    all_series = {}

    for ticker, name in FRED_TICKERS.items():
        print(f"  {ticker} ({name})...", end=" ")

        try:
            s = fred.get_series(ticker, observation_start=start, observation_end=end)
            s.name = name

            # Save individual CSV
            s.to_csv(os.path.join(out_dir, f"{ticker}.csv"), header=True)

            all_series[name] = s
            print(f"OK ({len(s)} obs)")

        except Exception as e:
            print(f"FAILED: {e}")

    # Save combined CSV
    combined = pd.DataFrame(all_series)
    combined.index.name = "date"

    combined.to_csv(os.path.join(out_dir, "usdyc_combined.csv"))

    print(f"\nSaved combined file: {out_dir}/usdyc_combined.csv")
    print(f"Shape: {combined.shape}")
    print(f"Date range: {combined.index[0]} → {combined.index[-1]}")

    print("\nFirst 5 rows:")
    print(combined.head())

    print("\nMissing values per tenor:")
    print(combined.isnull().sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2023-02-16")
    parser.add_argument("--out_dir", default="data/raw")

    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")

    # ✅ Debug (you can remove later)
    print("DEBUG: API KEY =", api_key)

    if not api_key:
        print("\nERROR: FRED_API_KEY not set.")
        print("Make sure .env exists in project root (iqf_project/.env)")
        print("Example:\nFRED_API_KEY=your_key_here\n")
        exit(1)

    download_fred(api_key, args.start, args.end, args.out_dir)
