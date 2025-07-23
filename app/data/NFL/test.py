import pandas as pd
from pathlib import Path

# Define the data directory using pathlib
DATA_DIR = Path(__file__).parent  # This will point to ...\app\data\NFL

# List of CSV files to merge
files = [
    "team_historical_data_2010_to_2020.csv",
    "team_historical_data_2021.csv",
    "team_historical_data_2022.csv",
    "team_historical_data_2023.csv",
    "team_historical_data_2024.csv",
    "team_historical_data_2025.csv"
]

print(f"📂 Looking for files in: {DATA_DIR.resolve()}")
print("🕵️‍♂️ Files present:", [f.name for f in DATA_DIR.glob("*.csv")])

dfs = []
missing_files = []
for file in files:
    file_path = DATA_DIR / file
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"✅ Loaded: {file}")
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")
    else:
        print(f"❌ Missing file: {file}")
        missing_files.append(file)

if not dfs:
    print("🚫 No files loaded. Exiting.")
else:
    # Concatenate and drop duplicates
    merged_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    output_file = DATA_DIR / "head_to_head.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"🎯 Merged {len(dfs)} files into {output_file.name} with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns!")
    if missing_files:
        print("⚠️ Some files were missing and not merged:", missing_files)
