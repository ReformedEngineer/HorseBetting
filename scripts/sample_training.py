import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load horse racing data from a CSV file
data = pd.read_csv('horse_racing_data.csv')

# Encode categorical features using LabelEncoder
encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']):
    data[col] = encoder.fit_transform(data[col])

# Scale numerical features using StandardScaler
scaler = StandardScaler()
for col in data.select_dtypes(include=['float64']):
    data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

# Group the data by horse
grouped_data = data.groupby('Horse')

# Train one neural network for each horse
models = {}
for name, group in grouped_data:
    X = group.drop(['Horse', 'Finish_Time'], axis=1)
    y = group['Finish_Time']

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, batch_size=32, verbose=1)

    models[name] = model

# Make predictions for each horse using its corresponding model
predictions = []
for name, group in grouped_data:
    X = group.drop(['Horse', 'Finish_Time'], axis=1)
    model = models[name]
    y_pred = model.predict(X)
    predictions.extend(y_pred)

# Calculate the mean squared error
y_test = data['Finish_Time']
mse = np.mean((y_test - predictions) ** 2)
print(f'Mean Squared Error: {mse}')
