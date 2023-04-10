import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os
current_script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(current_script_path)
project_root = os.path.dirname(script_directory)
sys.path.append(project_root)
from utils import encode_categorical_features, create_embedding_model,split_inputs
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Concatenate, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
tf.config.threading.set_inter_op_parallelism_threads(16)
tf.config.threading.set_intra_op_parallelism_threads(16)
import pickle


# Load the saved preprocessing objects
with open("models/generic/preprocessing_objects.pkl", "rb") as f:
    preprocessing_objects = pickle.load(f)

encoders = preprocessing_objects["encoders"]
scaler = preprocessing_objects["scaler"]
y_scaler = preprocessing_objects["y_scaler"]


model = models.load_model("models/generic/generic.h5")

# Read the dataset you want to make predictions on
data_to_predict = pd.read_csv("Train_specific.csv")
# data_to_predict = data_to_predict.drop(["horse"], axis=1)


# Apply the same preprocessing steps as you did for the training data
categorical_features = [
    'RaceTime', 
    'Race', 
    'Type', 
    # 'Class', 
    'AgeLimit',
    # 'Ran',
    # 'Yards', 
    # 'Seconds', 
    # 'Sp',
    # 'Age', 
    # 'WeightLBS', 
    'Trainer', 
    'Jockey', 
    'Going 2', 
    'Course2', 
    'Region2'
]

numerical_features = [
    'Class', 
    'Yards',
    'Ran',
    'Sp',
    'Age', 
    'WeightLBS',
]

data_to_predict[categorical_features] = data_to_predict[categorical_features].astype('str')
data_to_predict[categorical_features] = data_to_predict[categorical_features].applymap(str.lower)
data_to_predict[numerical_features] = data_to_predict[numerical_features].apply(pd.to_numeric, errors='coerce')
data_to_predict['Class'] = data_to_predict['Class'].fillna(0)
print(data_to_predict.shape)
# Remove rows with unseen labels in categorical features
for feature in categorical_features:
    data_to_predict = data_to_predict[data_to_predict[feature].isin(encoders[feature].classes_)]

print(data_to_predict.shape)

# Encode categorical features using the same encoders used for training 
for feature in categorical_features:
    data_to_predict[feature] = encoders[feature].transform(data_to_predict[feature])

# Scale the numerical features using the same scaler used for training
data_to_predict[numerical_features] = scaler.transform(data_to_predict[numerical_features])

# Make predictions
predictions = model.predict(split_inputs(data_to_predict, categorical_features, numerical_features))

# Inverse transform the predictions using the y_scaler used for training
predictions = y_scaler.inverse_transform(predictions)

# # Print the predictions
# print(predictions)

data_to_predict['predictions'] = predictions

# Create a new column 'totalbtn_rank' with rankings based on 'totalbtn' in ascending order
data_to_predict['totalbtn_rank'] = data_to_predict.groupby('Id')['TotalBtn'].rank(ascending=True)

# Create a new column 'predictions_rank' with rankings based on 'predictions' in ascending order
data_to_predict['predictions_rank'] = data_to_predict.groupby('Id')['predictions'].rank(ascending=True)

data_to_predict['sp_rank'] = data_to_predict.groupby('Id')['Sp'].rank(ascending=True)

# Inverse transform the numerical features in data_to_predict
data_to_predict[numerical_features] = scaler.inverse_transform(data_to_predict[numerical_features])

# Inverse transform the categorical features in data_to_predict
for feature in categorical_features:
    data_to_predict[feature] = encoders[feature].inverse_transform(data_to_predict[feature])


# Save the DataFrame with predictions as a new CSV file
data_to_predict.to_csv('data_with_predictions.csv', index=False)

# data_to_predict['Seconds'] = pd.to_numeric(data_to_predict['Seconds'], errors='coerce')
# data_to_predict['predictions'] = pd.to_numeric(data_to_predict['predictions'], errors='coerce')

# min_secs_idx = data_to_predict.groupby('race_name')['Seconds'].idxmin()
# min_predictions_idx = data_to_predict.groupby('race_name')['predictions'].idxmin()

# # Check how many of the minimum 'secs' and 'predictions' indices are the same
# correct_predictions_first_place = sum(min_secs_idx == min_predictions_idx)

# max_secs_idx = data_to_predict.groupby('race_name')['secs'].idxmax()
# max_predictions_idx = data_to_predict.groupby('race_name')['predictions'].idxmax()

# # Check how many of the minimum 'secs' and 'predictions' indices are the same
# correct_predictions_last_place = sum(max_secs_idx == max_predictions_idx)

# # Calculate the total number of unique races
# total_races = len(data_to_predict['race_name'].unique())

# # Print the number of correct predictions and the total number of races
# print(f"Correct predictions first place: {correct_predictions_first_place}/{total_races}")
# print(f"Correct predictions last place: {correct_predictions_last_place}/{total_races}")

# # Get the top 3 predictions for each race and course
# top3_predictions = data_to_predict.groupby(['race_name', 'course'])['predictions'].nsmallest(3)

# # Get the top 3 actual finishers for each race and course
# top3_secs = data_to_predict.groupby(['race_name', 'course'])['secs'].nsmallest(3)

# # Find the correct predictions
# correct_predictions_1 = 0
# correct_predictions_2 = 0
# correct_predictions_3=0
# for (race_name, course) in data_to_predict[['race_name', 'course']].drop_duplicates().to_numpy():
#     try:
#         correct_count = len(set(top3_predictions[(race_name, course)].index).intersection(set(top3_secs[(race_name, course)].index)))
#     except KeyError:
#         correct_count = 0
#     if correct_count >= 1:
#         correct_predictions_1 += 1
#     if correct_count >= 2:
#         correct_predictions_2 += 1
#     if correct_count >= 3:
#         correct_predictions_3 += 1

# # Calculate the total number of unique races and courses
# total_races_courses = len(data_to_predict[['race_name', 'course']].drop_duplicates())

# # Print the number of correct predictions and the total number of races and courses
# print(f"Correct top 3 predictions (at least 1 in top 3): {correct_predictions_1}/{total_races_courses}")
# print(f"Correct top 3 predictions (at least 2 in top 3): {correct_predictions_2}/{total_races_courses}")
# print(f"Correct top 3 predictions (all 3 in top 3): {correct_predictions_3}/{total_races_courses}")

# Filter the DataFrame to only include rows where the predicted rank and actual rank are equal
correct_predictions = data_to_predict[data_to_predict['predictions_rank'] == data_to_predict['totalbtn_rank']]

# Count the occurrences of ranks 1, 2, and 3
correct_1 = (correct_predictions['predictions_rank'] == 1).sum()
correct_2 = (correct_predictions['predictions_rank'] == 2).sum()
correct_3 = (correct_predictions['predictions_rank'] == 3).sum()


# Filter the DataFrame to only include rows where the predicted rank and actual rank are both 1
correct_rank_1 = data_to_predict[(data_to_predict['predictions_rank'] == 1) & (data_to_predict['totalbtn_rank'] == 1)]
bets_to_take = data_to_predict[(data_to_predict['predictions_rank'] == 1) & (data_to_predict['sp_rank']!=1)]
horse_won= bets_to_take[(bets_to_take['predictions_rank'] == 1) & (bets_to_take['totalbtn_rank'] == 1)]
# bets_to_take.to_csv('bets.csv')
num_bets_to_take = len(bets_to_take)
print(f"Number of bets to take: {num_bets_to_take} and Money won : {horse_won['Sp'].sum()}")
# better_rank_1 = correct_rank_1[correct_rank_1['predictions_rank'] < correct_rank_1.sp_rank]

# Calculate the sum of the 'Sp' column for the filtered DataFrame
sp_sum_rank_1 = correct_rank_1['Sp'].sum()


Totalraces= len(data_to_predict.Id.unique())

print(f"Number of correct predictions for rank 1: {correct_1} out of {Totalraces} races and revenue of {sp_sum_rank_1}")
print(f"Number of correct predictions for rank 2: {correct_2} out of {Totalraces} races")
print(f"Number of correct predictions for rank 3: {correct_3} out of {Totalraces} races")


# Define variables to count the number of races with at least 1 and 2 horses in the top 3
at_least_1_correct = 0
at_least_2_correct = 0

# Group the DataFrame by the 'id' column
grouped_data = data_to_predict.groupby('Id')

for group_id, group_df in grouped_data:
    # Filter the DataFrame to only include rows where the predicted rank is in the top 3
    top_3_predictions = group_df[group_df['predictions_rank'] <= 3]

    # Count the number of predictions with their actual rank in the top 3
    correct_predictions = (top_3_predictions['totalbtn_rank'] <= 3).sum()

    if correct_predictions >= 1:
        at_least_1_correct += 1

    if correct_predictions >= 2:
        at_least_2_correct += 1

print(f"Number of races with at least 1 horse in top 3 predictions: {at_least_1_correct} out of {Totalraces} races")
print(f"Number of races with at least 2 horses in top 3 predictions: {at_least_2_correct} out of {Totalraces} races")