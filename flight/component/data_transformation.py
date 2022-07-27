from flight.entity.artifact_entity import DataIngestionArtifact, \
    DataTransformationArtifact, DataValidationArtifact
from flight.entity.config_entity import DataTransformationConfig
from flight.logger import logging
from flight.exception import FlightException

import os,sys
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd

from flight.constant import *
from flight.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data
    

# Date_of_Journey
# Dep_Time
# Arrival_Time
# Duration

# Airline
# Source
# Destination
# Total_Stops

class FeatureSplitter(BaseEstimator, TransformerMixin):
    
    def __init__ (self,date_columns : list =[0], time_columns : list = [1,2],
                 sep = "_"):
        """
        date_columns - 
        Date_of_Journey idx = 0

        time_columns - 
        Dep_Time idx = 1
        Arrival_Time idx = 2

        """
        try:
            self.date_columns : list = date_columns
            self.time_columns : list = time_columns
            self.sep = sep

        except Exception as e:
            raise FlightException(e,sys) from e

    def fit (self,X, y=None):
        return self

    def transform(self,X, y=None):
        try:
            # convert any dataframe to numpy array
            if type(X) is not np.ndarray:
                X = X.to_numpy()

            # date_columns spliting
            for col in self.date_columns:
                feat_day = pd.to_datetime(X[:,col]).day
                feat_month = pd.to_datetime(X[:,col]).month
                X = np.c_[X,feat_day,feat_month]

            # time_columns splitting
            for col in self.time_columns:
                feat_hour = pd.to_datetime(X[:,col]).hour
                feat_minute = pd.to_datetime(X[:,col]).minute
                X = np.c_[X,feat_hour,feat_minute]

            # delete the redundant columns
            idx_shift = 0
            for col in self.date_columns + self.time_columns:
                X = np.delete(X,col-idx_shift,axis=1)
                idx_shift += 1

            return X
        except Exception as e:
            raise FlightException(e,sys) from e
        
class FeatureCalculator(BaseEstimator, TransformerMixin):
    
    def __init__ (self,feat_calculate_idx = [0]):
        """
        feat_delete_idx - 
        Route idx = 0
        Additional_info idx = 2

        feat_calculate idx - 
        Duration idx = 2
        """
        try:
            self.feat_calculate_idx : list = feat_calculate_idx

        except Exception as e:
            raise FlightException(e,sys) from e 

    def fit (self,X, y=None):
        return self

    def transform(self,X, y=None)-> np.array:
        try:
            # convert any dataframe to numpy array
            if type(X) is not np.ndarray:
                X = X.to_numpy()

            idx_shift = 0
            
            """# delete the req. Features
            for col in self.feat_delete_idx:
                X = np.delete(X,col-idx_shift,axis=1)
                idx_shift += 1"""

            # calculate the feature value
            for col in self.feat_calculate_idx:
                feat = list(X[:,col-idx_shift])
                # Check if feat contains only hour or mins
                for i in range(len(feat)):
                    if len(feat[i].split()) != 2:    
                        if "h" in feat[i]:
                            feat[i] = feat[i].strip() + " 0m"   # Adds 0 minute
                        else:
                            feat[i] = "0h " + feat[i]   # Adds 0 hour

                feat_hours = []
                feat_minutes = []
                    
                # Extract hours & minutes
                for i in range(len(feat)):
                    feat_hours.append(int(feat[i].split(sep = "h")[0]))    
                    feat_minutes.append(int(feat[i].split(sep = "m")[0].split()[-1]))    
                X = np.c_[X,feat_hours,feat_minutes]
                # delete the redundant column
                X = np.delete(X,col-idx_shift,axis=1)
                idx_shift += 1

            return X

        except Exception as e:
            raise FlightException(e,sys) from e


class DataTransformation:
    
    def __init__(self,data_transformation_config:DataTransformationConfig,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20} ")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_nominal_columns = dataset_schema[CATEGORICAL_NOMINAL_COLUMN_KEY]
            categorical_ordinal_columns = dataset_schema[CATEGORICAL_ORDINAL_COLUMN_KEY]

            num_pipeline = Pipeline(steps=[
                ("feature_splitter", FeatureSplitter()),
                ("feature_calculator", FeatureCalculator()),
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ]
            )

            cat_ohe_pipeline = Pipeline(steps=[
                ("One_Hot_Encoder",OneHotEncoder(drop="first",sparse=False)),
                ("simple_imputer", SimpleImputer(strategy="most_frequent")),
                ("scalar", StandardScaler(with_mean=False))
            ]
            )
            
            cat_ordinal_pipeline = Pipeline(steps=[
                ("simple_imputer", SimpleImputer(strategy="most_frequent")),
                ("Ordinal_encoder",OrdinalEncoder()),
                ("scalar", StandardScaler(with_mean=False))
            ]
            )
                        
            logging.info(f"Categorical Nominal columns: {categorical_nominal_columns}")
            logging.info(f"Categorical Ordinal columns: {categorical_ordinal_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_ohe_pipeline', cat_ohe_pipeline, categorical_nominal_columns),
                ('cat_ordinal_pipeline', cat_ordinal_pipeline, categorical_ordinal_columns)
            ])
            
            return preprocessing

        except Exception as e:
            raise FlightException(e,sys) from e   

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training  and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # concatinating input features and target variable
            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed=True,
                message="Data transformation successfull.",
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_obj_file_path
            )
            
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")