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

# Read the dataset
data = pd.read_csv("Train_general.csv")

# Encode categorical features
categorical_features = [
    'type',
    'age_band',
    # 'dist_m'
    'going',
    # 'ran'
    # 'horse',
    # 'age'
    'sex',
    # 'lbs'
    # 'secs'
    'jockey',
    'trainer',
    'owner',
    'race_name',
    # 'Unique_id'
]

numerical_features = [
    'dist_m',
    'ran',
    'age',
    'lbs',
]

data[categorical_features] = data[categorical_features].astype('str')

# # Select all non-categorical columns
# numerical_features = data.select_dtypes(exclude='object').columns


# Force-convert non-categorical columns to numeric data types
data[numerical_features] = data[numerical_features].apply(pd.to_numeric, errors='coerce')

data['secs']= data['secs'].apply(pd.to_numeric, errors='coerce')

data=data.dropna()
encoded_data, encoders = encode_categorical_features(data, categorical_features)

# Split the dataset into features and target variable
X = encoded_data.drop("secs", axis=1)
y = encoded_data["secs"]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val= y_scaler.transform(y_val.values.reshape(-1, 1))


with open("preprocessing_objects.pkl", "wb") as f:
    pickle.dump({"encoders": encoders, "scaler": scaler, "y_scaler": y_scaler}, f)
    

print("Missing values in X_train:", X_train.isna().sum().sum())
print("Missing values in X_val:", X_val.isna().sum().sum())
print("Missing values in y_train:", np.isnan(y_train).sum())
print("Missing values in y_val:", np.isnan(y_val).sum())

print("Infinite values in X_train:", np.isinf(X_train).sum().sum())
print("Infinite values in X_val:", np.isinf(X_val).sum().sum())
print("Infinite values in y_train:", np.isinf(y_train).sum())
print("Infinite values in y_val:", np.isinf(y_val).sum())

# Determine the number of unique categories for each categorical feature
category_sizes = {feature: len(encoders[feature].classes_) for feature in categorical_features}


# Create the embedding model
embedding_model = create_embedding_model(category_sizes, len(numerical_features))

early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# # Train the model
embedding_model.fit(split_inputs(X_train, categorical_features, numerical_features), 
                    y_train, epochs=100, validation_data=(split_inputs(X_val, categorical_features, numerical_features), y_val),
                    callbacks=[early_stopping],verbose=1)

embedding_model.save("generic.h5")