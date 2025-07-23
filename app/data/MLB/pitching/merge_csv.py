import pandas as pd
import glob
import os

# Set the directory containing your CSV files
csv_dir = 'app/data/MLB/pitching'
# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

# Read and concatenate all CSV files
df_list = [pd.read_csv(f) for f in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Save to a single CSV file
combined_df.to_csv(os.path.join(csv_dir, 'pitching_data_combined.csv'), index=False)