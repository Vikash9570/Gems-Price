import sys
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
import os
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer



@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_taransformed_object(self):
        try:
            logging.info("Data Transform Initiated")
            # defining which column should be ordinal-encoded and which should be scaled
            categorical_col=['cut', 'color','clarity']
            numerical_col=['carat', 'depth','table', 'x', 'y', 'z']

            # defining custom rank for each ordinal encoder
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("pipeline initiated")

            # numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            # catagorical pipeline
            cat_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(strategy="most_frequent")),
                       ("ordinalencoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                       ("scaler",StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_col),
                ("cat_pipeline",cat_pipeline,categorical_col)
            ])

            return preprocessor
        
            logging.info("pipeline completed")

        except Exception as e:
            logging.info("error in data DataTransformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
        # reading dataset
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            

            logging.info(f'cgecking shape of data{train_df.shape,test_df.shape}')

            logging.info("read train and test data completes")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info("obtaining perprocessor object")
            preprocessing_obj=self.get_data_taransformed_object()

            target_column_name="price"
            drop_columns=[target_column_name,"id"]

            logging.info("getting section of datasets")
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            # input_feature_train_df=input_feature_train_df.iloc[ :15000, : ]

            target_feature_train_df=train_df[target_column_name]
            # target_feature_train_df=target_feature_train_df.iloc[:15000,:]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            # input_feature_test_df=input_feature_test_df.iloc[ :15000,: ]
            target_feature_test_df=test_df[target_column_name]
            # target_feature_test_df=target_feature_test_df.iloc[:15000,:]

            ## transforming using preprocessor object
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on train and test dataset ")

            # concatenating train and test array with there target features
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info("preprocessor pickel file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("error in data DataTransformation")
            raise CustomException(e,sys)
        



# checking it in data ingestion becz ingestion first after transformation





         




























