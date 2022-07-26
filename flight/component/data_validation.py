import json
from pydoc import html
from flight.constant import DATA_DRIFT_DATA_DRIFT_KEY, DATA_DRIFT_DATA_KEY, \
    DATA_DRIFT_DATASET_DRIFT_KEY, DATA_DRIFT_METRICS_KEY, SCHEMA_COLUMNS_KEY, SCHEMA_DOMAIN_VALUE_KEY
from flight.entity.config_entity import DataValidationConfig
from flight.logger import logging
from flight.exception import FlightException
import os,sys
from flight.entity.artifact_entity import DataIngestionArtifact,\
    DataValidationArtifact
from flight.util.util import read_yaml_file,save_json_file,load_data

import pandas as pd
import numpy as np

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab


class DataValidation:
    
    def __init__(self,data_validation_config: DataValidationConfig,
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20} ")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

        except Exception as e:
            raise FlightException(e,sys) from e
        
    def get_train_and_test_dataset (self):
        try:
            # get the data sets
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logging.info(f"Train & Test Dataset loaded")
            return train_df,test_df     
        
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def check_train_and_test_file_exists (self)->bool:
        try:
            data_exists = False
            
            # get the data file paths
            train_file_path:str = self.data_ingestion_artifact.train_file_path
            test_file_path:str = self.data_ingestion_artifact.test_file_path
            
            # check if the file paths are valid
            if os.path.exists(train_file_path) and os.path.exists(test_file_path):
                data_exists = True
            else:
                message = f"Train path : [{train_file_path}] \n Test path : [{test_file_path}] \n Check : One or Both are Missing"
                raise Exception(message)
            
            logging.info(f"Train & Test data Check is Completed \n Train path : [{train_file_path}] \n Test path : [{test_file_path}]")    
            return data_exists

        except Exception as e:
            raise FlightException(e,sys) from e
        
    def validate_num_columns(self)->bool:
        try:
            validate_num_columns = False
            no_col_schema = len(self.schema[SCHEMA_COLUMNS_KEY])
            no_col_train_df = len(self.train_df.columns)
            no_col_test_df = len(self.test_df.columns)
            if (no_col_schema==no_col_train_df) and (no_col_schema==no_col_test_df):
                validate_num_cols = True
            
            logging.info(f"Number of Columns Check: Passed")
            return validate_num_columns                
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def validate_column_names(self)->bool:
        try:
            validate_column_names = False
            for column in self.train_df.columns:
                if not column in self.schema[SCHEMA_COLUMNS_KEY]:
                    message = f"{column} not in schema file"
                    raise Exception(message)
            validate_column_names = True
            
            logging.info(f"Column Names Check: Passed")
            return validate_column_names
            
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def validate_domain_values (self)->bool:
        try:
            validate_domain_values = False
            for column,category_list in self.schema[SCHEMA_DOMAIN_VALUE_KEY]. \
                items():
                for category in self.train_df[column].unique():
                    if category not in category_list and category is not np.nan:
                        message = f"[{column}] column does not accept <{category}> in schema"
                        raise Exception(message)
            validate_domain_values = True
            
            logging.info(f"Domain Values Check: Passed")    
            return validate_domain_values
        
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def validate_column_dtypes (self):
        try:
            validate_column_dtypes = False
            for column in self.train_df.columns:
                try:
                    schema_dtype = self.schema[SCHEMA_COLUMNS_KEY][column]
                    column_dtype = str(self.train_df[column].dtype)
                    if not column_dtype == schema_dtype:
                        self.train_df[column].astype(schema_dtype)
                except Exception as e:
                    message = f"[{column}] : dtype [{column_dtype}] " + \
                    "\n {column_dtype} cannot be typecasted to <{schema_dtype}>"
                    raise Exception(message)
            
            validate_column_dtypes = True
            
            logging.info(f"Column Dtypes Check: Passed")
            return validate_column_dtypes      
              
        except Exception as e:
            raise FlightException(e,sys) from e
        
        
    def validate_schema (self)->bool:
        try:
            is_validated = False
            schema_file_path = self.data_validation_config.schema_file_path
            # read the schema
            self.schema = read_yaml_file(file_path=schema_file_path)
            
            # get the train & test data
            
            self.train_df = load_data(file_path=self.data_ingestion_artifact.train_file_path,\
                schema_file_path=schema_file_path)
            self.test_df = load_data(file_path=self.data_ingestion_artifact.test_file_path,\
                schema_file_path=schema_file_path)        
            
            #1. Number of Column
            validated_num_columns = self.validate_num_columns()
            #2. Check column names
            validated_column_names = self.validate_column_names()
            #3. Check the value of ocean proximity 
            validated_domain_values = self.validate_domain_values()
            #4. check dtypes of columns
            validated_column_dtypes = self.validate_column_dtypes()
            
            is_validated =  (validated_num_columns & validated_column_names 
                            & validated_domain_values & validated_column_dtypes)
            
            logging.info(f"Validation of Schema is Completed")
            return is_validated
        
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def get_and_save_data_drift_report_file(self)->json:
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            profile.calculate(self.train_df,self.test_df)
            report = profile.json()
            
            report_json = json.loads(report)
            save_json_file(
                file_path = self.data_validation_config.report_file_path,
                file=report_json
                )
            
            logging.info(f"JSON Report has been generated successfully.")
            return report_json
            
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def save_data_drift_report_page_file(self)->html:
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            dashboard.calculate(self.train_df,self.test_df)
            
            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_file_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_file_dir,exist_ok=True)
            
            dashboard.save(report_page_file_path)
            logging.info(f"HTML Report has been generated " + \
                "\n {os.path.basename(report_page_file_path)} located in " + \
                "{report_page_file_path}")
            
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def check_data_drift (self)->bool:
        try:
            validated_data_drift = False
            report = self.get_and_save_data_drift_report_file()
            if report[DATA_DRIFT_DATA_DRIFT_KEY][DATA_DRIFT_DATA_KEY][
                DATA_DRIFT_METRICS_KEY][DATA_DRIFT_DATASET_DRIFT_KEY]:
                message = f"Data Drift is found in Dataset"
                raise Exception(message)
            self.save_data_drift_report_page_file()       
            validated_data_drift=True
                
            logging.info(f"Data Drift Check: Passed")
            return validated_data_drift
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            is_train_and_test_file_exists = self.check_train_and_test_file_exists()
            is_validated:bool = self.validate_schema()
            is_data_drift = self.check_data_drift()
            
            schema_file_path = self.data_validation_config.schema_file_path
            report_file_path = self.data_validation_config.report_file_path
            report_page_file_path = self.data_validation_config.report_page_file_path
            message = f"Data Validation performed successully."
            
            data_validation_artifact = DataValidationArtifact(
                schema_file_path=schema_file_path,
                report_file_path=report_file_path,
                report_page_file_path=report_page_file_path,
                is_validated=is_validated,
                message=message
            )
            
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")