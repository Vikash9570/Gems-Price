# basic import
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.utils import save_object
from src.exception import CustomException


from src.logger import logging
from dataclasses import dataclass
from src.utils import evaluate_model
import sys
import os


from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("splitting of dependent and independent variable from train and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models={
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "ElasticNet":ElasticNet()
            }

            model_report=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n==============================================================\n")
            logging.info(f'model report :{model_report}')

            # to get best model score from dictionary
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            print(f'best model name :{best_model_name} and model R2 score is :{best_model_score}')
            print("\n==========================================================================\n")
            logging.info(f'best model name :{best_model_name} and model R2 score is :{best_model_score}')
            

            best_model=models[best_model_name]

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )



        except Exception as e:
            logging.info("error occured at model training")
            raise CustomException(e,sys)









