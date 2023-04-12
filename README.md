# Instructons to Reproduce & work with the Repo

## Project Diagram 
![](process.png)

## Clone the repo
Start by cloning the repo on your local machine by running
1. First, the collaborator needs to create a personal access token on the platform where the repository is hosted (e.g., GitHub, GitLab). The token should have the necessary permissions (such as "repo" scope for GitHub) to access and interact with private repositories.  
    a. Go to Settings > Developer settings > Personal access tokens.  
    b. Click "Generate new token."  
    c. Give the token a name (e.g., "Private Repo Access").  
    d. Select the necessary permissions (e.g., check the "repo" scope box).  
    e. Click "Generate token."  
    f. Copy the generated token, as it will not be shown again.
2. Run this command in your terminal with your gothb username and he token form above git clone https://username:personal-access-token@github.com/ReformedEngineer/HorseBetting.git 

## Install the requirements

> pip install -r requirements.txt  
Now you should be ab;e to run any file in the repo and reproduce your results 

## Descriptions of the .py files 

1. create_training_data.py -> This files takesin the training data and splits it into the datasets required for trianing models  
2. generic.py -> trains the generic model 
3. transferlearning.py -> Adds horse specific models 

Pls look into the content of the fiels to change the paramters according to your needs

## How to run .py files

Inside the scrits folder are the project files that are used for various purposes. e.g. craete datasets , build generic model etc

to run a specific file you need to use python in your terminal or you can create a ipynb and copy the contents of the script you want to run in the notebook and then run cell wise 

command to run the script in terminal will be of this type 
> python3 scripts/predict_generic_specific.py  

inside the files there are paths to datsets that you can change, for example the command above is used to run the predict_general_specific.py file that takes in the csv which has data we need predcitons for and outputs the resut as a csv too . as you can see below, its loads the model and then takes in an input for data_to_predict (Test.csv file).. to predcit new data you can make changes here , if oyour new data is in test1.csv change it accordingly  

```python
model = models.load_model("models/generic/generic.h5")

# Read the dataset you want to make predictions on
data_to_predict = pd.read_csv("Test.csv")
# data_to_predict = data_to_predict.drop(["horse"], axis=1)
```

The script then saves two datsets as output 
```python
# Save the DataFrame with predictions as a new CSV file
data_to_predict.to_csv('data_with_spec_predictions.csv', index=False)
horses_data= data_to_predict[data_to_predict['HorseName2'].str.lower().isin(specific_horses)]
horses_data.to_csv('Horses.csv', index=False)
```
## Next steps
The model predcicts TotalBtn for horses based on the trained models and produces an ok result. The model is in utils.py file and is shown below 

```python
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
```
To improve the base model is a continuous trial and refitting process and i can give some pointers how to go about it and what it usually involves 

1. **<mark>Train the model on more data<mark>** , for now its trained on 4 years of data from 2016 to 2020 for generic model . you can train it on 40 years of data of you ahve it. usually 4 years of data isnt a very big data.
2.  **<mark>Clean the dataset thoroughly<mark>** so that the model isnt leanring wrng things or applying them in a wrong way 
3.  **<mark>Hyperparameter Tuning<mark>** the model above is build on a basic architecture as seen in the models architecture above am using 12 and 7 layer nuerons with no logic simply becaue it was used in the paper. 

Hyper parameter tuning means you can define ranges of nuerons say from 0 to 200 in steps of 10 and a model will be built for each of these combinaiton so for two layers it will tr all combinations from 0 to 200 . 20 possibities each for both layers iplies 400 combinations to be build and tested.

You will then choose the combnation that gives best result. for example it can be 50 for first layer and 70 for next layer and you will retrain the generic and specific model with this to imporve the predcitions 

as you can see this is
1. appying the same logic as has bene coded for a sample architecture model so can be copied and implemented on your own
2. is extremeley time consuming as it will take hours and hours to run and then processed and then cleaned and re run again if needed

Summary is , to build a model is 30 hours of work to 35% correct predcitions to refine the model to take it to 50% predcition accuracy is a continuius and much longer work  