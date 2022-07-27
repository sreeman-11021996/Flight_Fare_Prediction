from collections import namedtuple
from datetime import datetime
from tkinter import EXCEPTION
import uuid
from flight.component.data_transformation import DataTransformation
from flight.component.model_pusher import ModelPusher
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
from flight.component.data_transformation import DataTransformation
from flight.component.model_trainer import ModelTrainer
from flight.component.model_evaluation import ModelEvaluation
from flight.component.model_pusher import ModelPusher

import os, sys
from collections import namedtuple
from datetime import datetime
import pandas as pd
from flight.constant import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME

Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp","artifact_time_stamp",
    "running_status", "start_time", "stop_time", "execution_time", "message","experiment_file_path", 
    "accuracy", "is_model_accepted"])

class Pipeline(Thread):
    experiment : Experiment = Experiment(*([None]*11))
    experiment_file_path = None
    
    def __init__(self,config: Configuration = Configuration()) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir,exist_ok=True)
            Pipeline.experiment_file_path = os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME,\
                EXPERIMENT_FILE_NAME)
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
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)\
        ->ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),\
                data_transformation_artifact=data_transformation_artifact)
            
            return model_trainer.initiate_model_trainer()
        
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def start_model_evaluation(self,model_trainer_artifact:ModelTrainerArtifact,\
        data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact)\
        ->ModelEvaluationArtifact:
        try:
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                model_trainer_artifact=model_trainer_artifact,
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact)
            
            return model_evaluation.initiate_model_evaluation()
        
        except Exception as e:
            raise FlightException(e,sys) from e

    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(model_pusher_config=self.config.get_model_pusher_config(),\
                model_evaluation_artifact=model_evaluation_artifact)
            
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise FlightException(e,sys) from e
        

    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment
            # data ingestion
            logging.info("Pipeline starting.")

            experiment_id = str(uuid.uuid4())
            start = datetime.now()
            
            Pipeline.experiment = Experiment(
                experiment_id=experiment_id,
                initialization_timestamp=self.config.time_stamp,
                artifact_time_stamp=self.config.time_stamp,
                running_status=True,
                start_time=start,
                stop_time=None,
                execution_time=None,
                message="Pipeline has been started.",
                experiment_file_path=Pipeline.experiment_file_path,
                accuracy=None,
                is_model_accepted=None)
            
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
            
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
            # model trainer
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact)
            # model evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact=model_trainer_artifact,
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            # model pusher
            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(
                    model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            else:
                logging.info("Trained model rejected.")
            logging.info("Pipeline completed.")
            
            stop = datetime.now()
            
            Pipeline.experiment = Experiment(
                experiment_id=experiment_id,
                initialization_timestamp=self.config.time_stamp,
                artifact_time_stamp=self.config.time_stamp,
                running_status=False,
                start_time=start,
                stop_time=stop,
                execution_time=start-stop,
                message="Pipeline has finished",
                experiment_file_path=Pipeline.experiment_file_path,
                accuracy=model_trainer_artifact.model_accuracy,
                is_model_accepted=model_evaluation_artifact.is_model_accepted)
            
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
            
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def save_experiment(self):
        try:
            pass
        except Exception as e:
            raise FlightException(e,sys) from e

