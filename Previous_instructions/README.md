# Horse Betting Project

## Table of Content

1. BaseMaterials -> this folder contains the material provided by you
2. data -> this folder contains data on years i have managed to scrape
3. scripts -> this folder contians base scripts built but needs modifiction as am understanding data better . this is where main work will be
4. Models -> this is where Models for prediciton and the parmatrs needed fo thos emodels lie

## Question /comment

1. In the paper provided the model is based on this (page 158 database section)

> Each horse race has a form which contains information
about the race and horses participating in the race [14].
In this paper one neural network have been used for each
horse, NN’s output is finishing time of the horse

this means if there are 40 horss racing in 2023, called Horse A, Horse B... Horce T etc .. there will be one model for each horse

this model will be trained by The speicif horses races

the problem is each horse may only have 3-4 races in ther career so this is not enough data to train

2.The authors acknowledges this on page 159 just above the diagram photo

> In this paper, we had some problems with data set. One
of the problems we had was about past performance of
horses. Some horses didn’t have enough past
performances as much as we need. For example, some
horses had only one or two past races. This history of a
horse was not adequate to predict the next race. So we
had to delete these horses from our data set, and
sometimes we had to shuffle past races of horses to
obtain good results.

## Solution

Authors hvent mentioned at all the way the splved this problem so am going to solve this using Transfer Learning

The final results may vary/ wil vary due to this fyi

## Methodology

### Training

1. Data from 2015 to 2020 will be used for training a general model using Back Propogation architecture. This model wil be built as close as posisble to the architecture described in the paper
2. Data for 2021 will then be used to fine tune this model on horse specific layers thus we will make one model for each horse that is running in 2022.. 

### Testing 

every horse thats still running and alive in 2022 will have its own model that will be used to predict its finsih time for 2022 Races. We will test these models for 2022 Races to get win percentage 

### Caveat

If there is a new horse in 2022 that didnt run any races in 2021, then there is no horse specific model for this Horse thus we wont be able to predcit sepcifically for this horse. we can use the general model for prediciton but keep in mind it ***wont be horse specific***

Same applies for a Horse in 2022 hat only ran 1 or 2 races in 2021

## Models

### Generic

The model trianed on data from 2015 to 2020 wihtout any horse specific details.In next step this model should be refined to show it the ata fro year 2021 for horses to craete horse specific models. 

This model is very rough right now as Data Cleaning is a very time consuming task and needs hours to properly look at each row of data. as an example racename @Ascot is the same as race name Ascot . due to imporper naming conventions across datasets , the model wont perform well so the data needs to be cleaned and made unform , i just take race name as an example to show the peopblem 

Yet the resukt of this model is ok for the first run , as can be seen by running scripts/predict_generic.py file resut shown for test data for year 2022 below . this data was never shown to the model and was compeletly new data . but this models needs a lot of data cleaning, (***this can be done by someone as its very time consiming task ??***)

> Correct predictions first place: 248/905  
Correct predictions last place: 253/905  
Correct top 3 predictions (at least 1 in top 3): 824/905  
Correct top 3 predictions (at least 2 in top 3): 545/905  
Correct top 3 predictions (all 3 in top 3): 167/905  

#### Steps to clean data and re produce the model 
1. Train_generic.csv is used for training model . Clean this file to the best of your requirement. Data clenaing is a task of going thorugh columns and removing the extra comma or dots or anything 
2. Same should be done for Test.csv which is 2022 data that we will use for testing the model results
3. once data is cleaned to your satisfaction , re run the scripts in this order  
    - generic.py this will train a new model called generic.h5 which will be saved in the models folder . this training step will take at least 2-3 hours to run . any library needed to be installed can be installed using pip3 install 'library'
    - then run predcit_generic.py to test the model on the Test.csv that must have been cleaned too 
    - the model lofgic and architecture remains same so no issue should be encoutered unless some columns have been deleted or added. 
     
