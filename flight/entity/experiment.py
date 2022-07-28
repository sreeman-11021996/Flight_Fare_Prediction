from flight.logger import logging
from flight.exception import FlightException
from flight.constant import *
from flight.config.configuration import Configuration

import os,sys
from collections import namedtuple
import uuid
import pandas as pd

Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp","artifact_time_stamp",
    "running_status", "start_time", "stop_time", "execution_time", "message","experiment_file_path", 
    "accuracy", "is_model_accepted"])

class Experiment_class:
    def __init__(self,*args):
        try:
            self.experiment : Experiment = Experiment(*args)
            
            # all are None
            self.experiment_id = args[0],
            self.initialization_timestamp=args[1],
            self.artifact_time_stamp=args[2],
            self.running_status=args[3],
            self.start_time=args[4],
            self.stop_time=args[5],
            self.execution_time=args[6],
            self.message=args[7],
            self.experiment_file_path = args[8]
            self.accuracy=args[9],
            self.is_model_accepted=args[10]
            
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def set_experiment_path(self,artifact_dir:str):
        try:
            os.makedirs(artifact_dir,exist_ok=True)
            self.experiment_file_path = os.path.join(artifact_dir,EXPERIMENT_DIR_NAME,EXPERIMENT_FILE_NAME)
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def check_pipeline_status(self)->Experiment:
        try:
            if self.experiment.running_status:
                logging.info("Pipeline is already running")
                return self.experiment
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def update_experiment(self):
        try:
            self.experiment = Experiment(
                experiment_id=self.experiment_id,
                initialization_timestamp=self.initialization_timestamp,
                artifact_time_stamp=self.artifact_time_stamp,
                running_status=self.running_status,
                start_time=self.start_time,
                stop_time=self.stop_time,
                execution_time=self.execution_time,
                message=self.message,
                experiment_file_path=self.experiment_file_path,
                accuracy=self.accuracy,
                is_model_accepted=self.is_model_accepted)
            
            logging.info(f"Pipeline experiment: {self.experiment}")
            self.save_experiment()
            
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def save_experiment(self):
        try:
            if self.experiment.experiment_id:
                experiment_dict = self.experiment._asdict()
                experiment_dict = {key:[value] for key,value in experiment_dict.items()}
                experiment_dict.update({
                    "created_time_stamp": [datetime.now()],
                    "experiment_file_path": [os.path.basename(self.experiment_file_path)]
                })
                
                experiment_report = pd.DataFrame.from_dict(experiment_dict)
                
                experiment_dir = os.path.dirname(self.experiment_file_path)
                os.makedirs(experiment_dir,exist_ok=True)
                # to write or append to experiment.csv
                if os.path.exists(self.experiment_file_path):
                    experiment_report.to_csv(self.experiment_file_path,index=False,header=False,mode="a")
                else:
                    experiment_report.to_csv(self.experiment_file_path,index=False,header=True,mode="w")
            else:
                print("First start experiment")
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def pipeline_started(self,time_stamp):
        try:
            logging.info("Pipeline starting.")
            
            self.experiment_id = str(uuid.uuid4())
            self.start_time = datetime.now()
            self.experiment_file_path = self.experiment_file_path
            self.running_status = True
            self.message = "Pipeline has been started."
            self.initialization_timestamp = time_stamp
            self.artifact_time_stamp = time_stamp
            
            self.update_experiment()
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def pipeline_ended(self,model_accuracy,is_model_accepted:bool):
        try:
            logging.info("Pipeline completed.")
            
            self.stop_time = datetime.now()
            self.running_status = False
            self.execution_time = self.stop_time - self.start_time
            self.message = "Pipeline has finished"
            self.accuracy = model_accuracy
            self.is_model_accepted = is_model_accepted
            
            self.update_experiment()            
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def get_experiment_status(self,limit:int = 5)->pd.DataFrame:
        try:
            if os.path.exists(self.experiment_file_path):
                experiment_df = pd.read_csv(self.experiment_file_path)
                limit = int(limit)
                return experiment_df.tail(limit).drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
            else:
                return pd.DataFrame()
        except Exception as e:
            raise FlightException(e,sys) from e
    
    