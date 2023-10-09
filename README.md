## End to End ML Project

### Info about Project 
```
In this Gems price prediction project dataset having attributes Carat,Cut,Color,Clarity,Depth,Table etc.
Price is target column, here pipeline has been created between diffrent MLOps and finally linked with front end to manually input data and get price as output. 

```

### created a environment
```
conda create -p venv python==3.10
```
### Install all necessary libraries
```
pip install -r requirements.txt
```
```
Steps for end to end project
```
## data ingestion
```
First of all Data is being ingested and splitting data into test and train dataframe

```
## data transformation
```
Attributes of dataset must be transformed to make dataset ready for fitting into model, this all things takecare in this stage.

```
## model training and generating scores  
```
with the help of pipeline data is fitted into model and get diffrent scores, based on good score model has been selected and pickle file is being saved in this stage.  

```