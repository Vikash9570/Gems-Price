import os
import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer













if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    obj_data_transformation=DataTransformation()
    train_arr,test_arr,_=obj_data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    obj_model_trainer=ModelTrainer()
    obj_model_trainer.initiate_model_training(train_arr,test_arr)










