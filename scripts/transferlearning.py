import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
from tensorflow.keras.optimizers import Adam


########################################################################
with open("models/generic/preprocessing_objects.pkl", "rb") as f:
    preprocessing_objects = pickle.load(f)

encoders = preprocessing_objects["encoders"]
scaler = preprocessing_objects["scaler"]
y_scaler = preprocessing_objects["y_scaler"]

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

df1= pd.read_csv('Train_specific.csv')
df2= pd.read_csv('Testing2.csv')

horses= common_horses(df1,df2,'HorseName2',12)



for horse in horses:
    
    # Load the base model for fine-tuning
    fine_tune_model = models.load_model("models/generic/generic.h5")

    # Set the layers to be trainable
    for layer in fine_tune_model.layers:
        layer.trainable = True
    
    # Read the specific horse dataset
    data =df1[df1.HorseName2.str.lower()==horse]
    data[categorical_features] = data[categorical_features].astype('str')
    data[categorical_features] = data[categorical_features].applymap(str.lower)
    
    data[numerical_features] = data[numerical_features].apply(pd.to_numeric, errors='coerce')

    # data['fn_distance']= data['fn_distance'].apply(pd.to_numeric, errors='coerce')
    data['Class'] = data['Class'].fillna(0)

    na_counts = data.isna().sum()
    
    print(na_counts)

    print(data.shape)
    data=data.dropna()
    print(data.shape)
    
    # Remove rows with unseen labels in categorical features
    for feature in categorical_features:
        data = data[data[feature].isin(encoders[feature].classes_)]

    print(data.shape)

    # Encode categorical features using the same encoders used for training 
    for feature in categorical_features:
        data[feature] = encoders[feature].transform(data[feature])

    # Scale the numerical features using the same scaler used for training
    data[numerical_features] = scaler.transform(data[numerical_features])

    # Split the dataset into features and target variable
    X_specific = data.drop("TotalBtn", axis=1)
    y_specific = data["TotalBtn"]
    y_specific = y_scaler.fit_transform(y_specific.values.reshape(-1, 1))
    
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_specific, y_specific, test_size=0.2, random_state=42)
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Fine-tune the model with the specific horse data
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    fine_tune_model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    fine_tune_model.fit(split_inputs(X_train, categorical_features, numerical_features), 
                    y_train, epochs=100, validation_data=(split_inputs(X_val, categorical_features, numerical_features), y_val),
                    callbacks=[early_stopping],verbose=1)

    fine_tune_model.save(f"models/specific/{horse}.h5")
    
    del fine_tune_model
    tf.keras.backend.clear_session()
    
    
    

