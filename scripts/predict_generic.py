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
data_to_predict = pd.read_csv("Test.csv")
data_to_predict = data_to_predict.drop(["horse"], axis=1)


# Apply the same preprocessing steps as you did for the training data
categorical_features = [
    'type',
    'age_band',
    'going',
    'sex',
    'jockey',
    'trainer',
    'owner',
    'race_name',
    'course',
]

numerical_features = [
    'dist_m',
    'ran',
    'age',
    'lbs',
]

data_to_predict[categorical_features] = data_to_predict[categorical_features].astype('str')
data_to_predict[numerical_features] = data_to_predict[numerical_features].apply(pd.to_numeric, errors='coerce')

print(data_to_predict.shape)
# Remove rows with unseen labels in categorical features
# for feature in categorical_features:
#     data_to_predict = data_to_predict[data_to_predict[feature].isin(encoders[feature].classes_)]

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

# Inverse transform the numerical features in data_to_predict
data_to_predict[numerical_features] = scaler.inverse_transform(data_to_predict[numerical_features])

# Inverse transform the categorical features in data_to_predict
for feature in categorical_features:
    data_to_predict[feature] = encoders[feature].inverse_transform(data_to_predict[feature])


# Save the DataFrame with predictions as a new CSV file
data_to_predict.to_csv('data_with_predictions.csv', index=False)

data_to_predict['secs'] = pd.to_numeric(data_to_predict['secs'], errors='coerce')
data_to_predict['predictions'] = pd.to_numeric(data_to_predict['predictions'], errors='coerce')

min_secs_idx = data_to_predict.groupby('race_name')['secs'].idxmin()
min_predictions_idx = data_to_predict.groupby('race_name')['predictions'].idxmin()

# Check how many of the minimum 'secs' and 'predictions' indices are the same
correct_predictions_first_place = sum(min_secs_idx == min_predictions_idx)

max_secs_idx = data_to_predict.groupby('race_name')['secs'].idxmax()
max_predictions_idx = data_to_predict.groupby('race_name')['predictions'].idxmax()

# Check how many of the minimum 'secs' and 'predictions' indices are the same
correct_predictions_last_place = sum(max_secs_idx == max_predictions_idx)

# Calculate the total number of unique races
total_races = len(data_to_predict['race_name'].unique())

# Print the number of correct predictions and the total number of races
print(f"Correct predictions first place: {correct_predictions_first_place}/{total_races}")
print(f"Correct predictions last place: {correct_predictions_last_place}/{total_races}")

# Get the top 3 predictions for each race and course
top3_predictions = data_to_predict.groupby(['race_name', 'course'])['predictions'].nsmallest(3)

# Get the top 3 actual finishers for each race and course
top3_secs = data_to_predict.groupby(['race_name', 'course'])['secs'].nsmallest(3)

# Find the correct predictions
correct_predictions_1 = 0
correct_predictions_2 = 0
correct_predictions_3=0
for (race_name, course) in data_to_predict[['race_name', 'course']].drop_duplicates().to_numpy():
    try:
        correct_count = len(set(top3_predictions[(race_name, course)].index).intersection(set(top3_secs[(race_name, course)].index)))
    except KeyError:
        correct_count = 0
    if correct_count >= 1:
        correct_predictions_1 += 1
    if correct_count >= 2:
        correct_predictions_2 += 1
    if correct_count >= 3:
        correct_predictions_3 += 1

# Calculate the total number of unique races and courses
total_races_courses = len(data_to_predict[['race_name', 'course']].drop_duplicates())

# Print the number of correct predictions and the total number of races and courses
print(f"Correct top 3 predictions (at least 1 in top 3): {correct_predictions_1}/{total_races_courses}")
print(f"Correct top 3 predictions (at least 2 in top 3): {correct_predictions_2}/{total_races_courses}")
print(f"Correct top 3 predictions (all 3 in top 3): {correct_predictions_3}/{total_races_courses}")