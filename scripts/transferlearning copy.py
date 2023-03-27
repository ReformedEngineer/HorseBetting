import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def encode_categorical_features(data, categorical_features):
    encoded_data = data.copy()
    encoders = {}

    for feature in categorical_features:
        encoder = LabelEncoder()
        encoded_data[feature] = encoder.fit_transform(encoded_data[feature])
        encoders[feature] = encoder

    return encoded_data, encoders

# Read the dataset
data = pd.read_csv("general_horses.csv")

# Encode categorical features
categorical_features = ["type_of_race", "horse_trainer", "horse_jockey", "track_condition", "weather"]
encoded_data, encoders = encode_categorical_features(data, categorical_features)

# Split the dataset into features and target variable
X = encoded_data.drop("finish_time", axis=1)
y = encoded_data["finish_time"]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
numerical_features = ["horse_weight", "number_of_horses", "race_distance"]
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])

# Next, create the model with embedding layers for the categorical features:

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Concatenate, Input, Flatten
from tensorflow.keras.models import Model

# Determine the number of unique categories for each categorical feature
category_sizes = {feature: len(encoders[feature].classes_) for feature in categorical_features}

# Define the embedding model architecture
def create_embedding_model(category_sizes, numerical_features_size):
    inputs = []
    embeddings = []

    # Create embedding layers for categorical features
    for feature, size in category_sizes.items():
        input_layer = Input(shape=(1,))
        embedding_layer = Embedding(input_dim=size, output_dim=int(size ** 0.5))(input_layer)
        flatten_layer = Flatten()(embedding_layer)

        inputs.append(input_layer)
        embeddings.append(flatten_layer)

    # Create input layer for numerical features
    numerical_input = Input(shape=(numerical_features_size,))
    inputs.append(numerical_input)
    embeddings.append(numerical_input)

    # Concatenate all embeddings and numerical features
    x = Concatenate()(embeddings)

    # Add dense layers
    x = Dense(5, activation="relu")(x)
    x = Dense(7, activation="relu")(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    return model

# Create the embedding model
embedding_model = create_embedding_model(category_sizes, len(numerical_features))


# Train the model using the preprocessed data:

def split_inputs(X):
    inputs = []
    for feature in X.columns:
        inputs.append(X[feature].values.reshape(-1, 1))
    return inputs

# Train the model
embedding_model.fit(split_inputs(X_train), y_train, epochs=100, validation_data=(split_inputs(X_val), y_val))
