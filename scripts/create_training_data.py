import glob
import os
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor
import time

def read_csv_file(csv_file,columns):
    df= pd.read_csv(csv_file)
    df= df[columns]
    return df

columns=[
    'date',
    # 'region'
    'course',
    # 'off'
    'race_name',
    'type',
    # 'class'
    # 'pattern'
    # 'rating_band'
    'age_band',
    # 'sex_rest'
    # 'dist'
    # 'dist_f'
    'dist_m',
    'going',
    'ran',
    # 'num'
    # 'pos'
    # 'draw'
    # 'ovr_btn'
    # 'btn'
    'horse',
    'age',
    'sex',
    'lbs',
    # 'hg'
    # 'time'
    'secs',
    # 'dec'
    'jockey',
    'trainer',
    # 'prize'
    # 'or'
    # 'rpr'
    # 'sire'
    # 'dam'
    # 'damsire'
    'owner',
    # 'comment'
]

input_path = "/home/am_nr/HorseBetting/data/dates/gb"  # Replace this with the path to your CSV files
output_Train_general = "Train_general.csv"  # Replace this with the desired output file name
output_Train_specific = "Train_specific.csv"  # Replace this with the desired output file name
output_Test = "Test.csv"  # Replace this with the desired output file name

# Find all CSV files in the input directory
csv_files = glob.glob(os.path.join(input_path, "*.csv"))

# # Read and concatenate all CSV files
# start_time = time.time()
# dfs = []
# for csv_file in csv_files:
#     df = pd.read_csv(csv_file)
#     df = df[columns]
#     dfs.append(df)
# sequential_time = time.time() - start_time

start_time = time.time()
with ProcessPoolExecutor() as executor:
    dfs = list(executor.map(read_csv_file, csv_files,itertools.repeat(columns)))

parallel_time = time.time() - start_time
# print(f"Sequential execution time: {sequential_time:.2f} seconds")
print(f"Parallel execution time: {parallel_time:.2f} seconds")

combined_df = pd.concat(dfs, axis=0, ignore_index=True)

print(combined_df.shape)

# combined_df['Unique_id'] = combined_df['date']+combined_df['race_name']
combined_df['date']= pd.to_datetime(combined_df['date'])
combined_df.sort_values(by='date', inplace=True) 
# print(combined_df.dtypes)

train_df_general= combined_df[combined_df.date<='2020']
train_df_specific= combined_df[(combined_df.date>'2020') & (combined_df.date<='2021' ) ]
test_df= combined_df[combined_df.date>'2021']
# Write the concatenated DataFrame to a new CSV file
train_df_general.drop(["date", "horse"], axis=1).to_csv(output_Train_general, index=False)
train_df_specific.drop(["date"], axis=1).to_csv(output_Train_specific, index=False)
test_df.drop(["date"], axis=1).to_csv(output_Test, index=False)