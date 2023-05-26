import sys, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj



'''By using the @dataclass without using the constructor __init__(), the class (DataTransformationConfig ) 
    accepted the value and assigned to the given variable, so that in this case automatically the preprocessor.pkl file will be created 
    in the artifacts folder'''
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is about data transformation
        '''
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy= 'median')),
                    ('scaling',StandardScaler())
                ]
            )
            logging.info("Standard scaling is completed")
            
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder()),
                    ('scaling',StandardScaler(with_mean=False))
                ]

            )
            logging.info("Categorical columns encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        return preprocessor


    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
            
                logging.info('Reading train-test data completed')
                logging.info('Reading train-test data completed')
                
                
                preprocessing_obj = self.get_data_transformer_object()
                target_column = 'math_score'
                
                input_features_train_df = train_df.drop([target_column],axis=1)
                target_feature_train_df = train_df[target_column]

                input_features_test_df = test_df.drop([target_column],axis=1)
                target_feature_test_df = test_df[target_column]

                logging.info('Applying preprocessor object on training and testing dataframe')

                input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
                input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

                train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
                

                test_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]

                logging.info("Saved preprocessing object.")
                save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                obj = preprocessing_obj
                        )


                return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                )
            except Exception as e:
                raise CustomException(e,sys)


