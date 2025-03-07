import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Concatenate, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

def encode_categorical_features(data, categorical_features):
    encoded_data = data.copy()
    encoders = {}

    for feature in categorical_features:
        encoder = LabelEncoder()
        encoded_data[feature] = encoder.fit_transform(encoded_data[feature])
        encoders[feature] = encoder

    return encoded_data, encoders

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

    # Add dense layers with Batch Normalization
    x = Dense(12, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(7, activation="relu")(x)
    x = BatchNormalization()(x)
    output_layer = Dense(1)(x)

    # Use a smaller learning rate and apply gradient clipping
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    # model.compile(loss='mean_squared_error', optimizer='adam')

    return model

# Train the model using the preprocessed data:

def split_inputs(X, categorical_features, numerical_features):
    inputs = []
    for feature in categorical_features:
        inputs.append(X[feature].values.reshape(-1, 1))
    inputs.append(X[numerical_features].values)
    return inputs

def common_horses(df1: pd.DataFrame, df2: pd.DataFrame, col:str, n:int) -> list:
    
    df1[col] = df1[col].str.lower()
    df2[col] = df2[col].str.lower()
    # Count the occurrences of each horse in df1
    horse_counts = df1[col].value_counts()

    # Filter df1 to only include horses that ran less than 5 races
    df1_filtered = df1[df1[col].isin(horse_counts[horse_counts > n].index)]
    # Find the values in the 'Horse' column that are present in both DataFrames
    common_values = pd.merge(df1_filtered[[col]], df2[[col]], on=col, how='inner')
    
    # Convert the common_values DataFrame to a list
    common_horses_list = common_values[col].unique().tolist()
    
    return common_horses_list
