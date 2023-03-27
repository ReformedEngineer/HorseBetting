import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read the dataset
data = pd.read_csv("general_horses.csv")

# Split the dataset into features and target variable
X = data.drop("finish_time", axis=1)
y = data["finish_time"]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


import tensorflow as tf
from tensorflow.keras import layers, models

# Define the base model architecture
def create_base_model():
    model = models.Sequential([
        layers.Dense(64, activation="relu", input_shape=(8,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Create and train the base model
base_model = create_base_model()
base_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))


# Save the base model
base_model.save("base_model.h5")

# Load the base model for fine-tuning
fine_tune_model = models.load_model("base_model.h5")

# Set the layers to be trainable
for layer in fine_tune_model.layers:
    layer.trainable = True


# Read the specific horse dataset
specific_data = pd.read_csv("specific_horse.csv")

# Split the dataset into features and target variable
X_specific = specific_data.drop("finish_time", axis=1)
y_specific = specific_data["finish_time"]

# Scale the features using the same scaler as before
X_specific = scaler.transform(X_specific)

# Fine-tune the model with the specific horse data
fine_tune_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
fine_tune_model.fit(X_specific, y_specific, epochs=50)
