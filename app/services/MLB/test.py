from app.config import DATA_DIR
import os
import pandas as pd


csv_dir = os.path.join(DATA_DIR, 'csv', "MLB")
os.makedirs(csv_dir, exist_ok=True)
path1 = os.path.join(csv_dir, 'mlb_historical_data(2020-2024).csv')
df1= pd.read_csv(path1)

path2 = os.path.join(csv_dir, 'mlb_historical_data(2017-2019).csv')
df2= pd.read_csv(path2)

path3 = os.path.join(csv_dir, 'mlb_historical_data(2014-2016).csv')
df3= pd.read_csv(path3)

path4 = os.path.join(csv_dir, 'mlb_historical_data(2010-2013).csv')
df4= pd.read_csv(path4)
# print("DataFrame shapes:", df1.shape, df2.shape, df3.shape, df4.shape)

# df = pd.concat([df1, df2, df3, df4], ignore_index=True)
# print("Combined DataFrame shape:", df.shape)
# df.to_csv(os.path.join(csv_dir, 'mlb_historical_data(2010-2024).csv'), index=False)
# print("Combined CSV file created successfully at:", os.path.join(csv_dir, 'mlb_historical_data(2010-2024).csv'))