# Horse Betting Project

## Table of Content
1. BaseMaterials -> this folder contains the material provided by you 
2. data -> this folder contains data on years i have managed to scrape 
3. scripts -> this folder contians base scripts built but needs modifiction as am understanding data better . this is where main work will be 

## Question /comment 

1. In the paper provided the model is based on this (page 158 database section)

> Each horse race has a form which contains information
about the race and horses participating in the race [14].
In this paper one neural network have been used for each
horse, NN’s output is finishing time of the horse

this means if there are 40 horss racing in 2023, called Horse A, Horse B... Horce T etc .. there will be one model for each horse 

this model will be trained by The speicif horses races 

the problem is each horse may only have 3-4 races in ther career so this is not enough data to train 

2. The authors acknowledges this on page 159 just above the diagram photo

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


