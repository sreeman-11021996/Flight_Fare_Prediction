from collections import namedtuple
from datetime import datetime
import uuid
from flight.component.data_transformation import DataTransformation
from flight.config.configuration import Configuration
from flight.logger import logging, get_log_file_name
from flight.exception import FlightException
from threading import Thread
from typing import List

from multiprocessing import Process
from flight.entity.artifact_entity import ModelPusherArtifact, DataIngestionArtifact, \
    ModelEvaluationArtifact,DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact
from flight.entity.config_entity import DataIngestionConfig, ModelEvaluationConfig

from flight.component.data_ingestion import DataIngestion
from flight.component.data_validation import DataValidation
#from flight.component.data_transformation import DataTransformation
#from flight.component.model_trainer import ModelTrainer
#from flight.component.model_evaluation import ModelEvaluation
#from flight.component.model_pusher import ModelPusher

import os, sys
from collections import namedtuple
from datetime import datetime
import pandas as pd
from flight.constant import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME

Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp","artifact_time_stamp",
    "running_status", "start_time", "stop_time", "execution_time", "message","experiment_file_path", 
    "accuracy", "is_model_accepted"])

class Pipeline(Thread):
    
    def __init__(self,config: Configuration = Configuration()) -> None:
        try:
            
            super().__init__(daemon=False, name="pipeline")
            self.config=config

        except Exception as e:
            raise FlightException(e,sys) from e
    
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.config.get_data_ingestion_config())
            
            return data_ingestion.initiate_data_ingestion()
        
        except Exception as e:
            raise FlightException(e,sys) from e    


    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)\
        ->DataValidationArtifact:
        try:
            data_validation = DataValidation(
                data_validation_config=self.config.get_data_validation_config(),
                data_ingestion_artifact=data_ingestion_artifact)
            
            return data_validation.initiate_data_validation()
        
        except Exception as e:
            raise FlightException(e,sys) from e

    def start_data_transformation(self,data_ingestion_artifact : DataIngestionArtifact,\
        data_validation_artifact : DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact)
            
            return data_transformation.initiate_data_transformation()
        
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise FlightException(e,sys) from e

    def start_model_trainer(self):
        pass

    def start_model_evaluation(self):
        pass

    def start_model_pusher(self):
        pass

    def run_pipeline(self):
        try:
            # data ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            # data validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact)
            # data transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            print(data_transformation_artifact)
            
        except Exception as e:
            raise FlightException(e,sys) from e

