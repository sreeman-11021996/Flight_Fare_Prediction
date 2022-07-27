
from flight.logger import logging
from flight.exception import FlightException
from flight.entity.config_entity import ModelEvaluationConfig
from flight.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,\
    ModelTrainerArtifact,ModelEvaluationArtifact
from flight.constant import *
import numpy as np
import os
import sys
from flight.util.util import write_yaml_file, read_yaml_file, load_object,load_data
from flight.entity.model_factory import MetricInfoArtifact, evaluate_regression_model

class ModelEvaluation:
    
    def __init__(self,model_evaluation_config:ModelEvaluationConfig,\
        model_trainer_artifact:ModelTrainerArtifact,data_ingestion_artifact:DataIngestionArtifact,
        data_validation_artifact:DataValidationArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def get_best_model_eval_file_path(self):
        try:
            # from model_evaluation.py
            best_model_file_path = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            
            # check if the yaml file exists - create if not exists
            if not os.path.exists(path=model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path)
                return best_model_file_path
            # check if there is content in the yaml file
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content
            
            if BEST_MODEL_KEY in model_eval_file_content:
                best_model_file_path = model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY]
                return best_model_file_path
        
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def update_evaluation_report(self,model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            # check if there is content in the yaml file
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            prev_best_model = None
            if BEST_MODEL_KEY in model_eval_file_content:
                prev_best_model = model_eval_file_content[BEST_MODEL_KEY]
            logging.info(f"Previous eval result: {model_eval_file_content}")
            # update the new best_model    
            eval_result = {
                BEST_MODEL_KEY : 
                    {MODEL_PATH_KEY : model_evaluation_artifact.evaluated_model_path}
                }
            # check if there is "history" or not
            if prev_best_model is not None:
                model_history = {
                    self.model_evaluation_config.time_stamp : {MODEL_PATH_KEY : prev_best_model}
                    }
                if HISTORY_KEY not in model_eval_file_content:
                    history = {HISTORY_KEY : model_history}
                    eval_result.update(history)
                else:
                    model_eval_file_content[HISTORY_KEY].update(model_history)
                
            model_eval_file_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_file_content}")
            write_yaml_file(file_path=model_evaluation_file_path,data=model_eval_file_content)
            
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            # curr. trained model
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)
            
            # get best model in production
            prod_model_file_path = self.get_best_model_eval_file_path()
            prod_model_object = load_object(file_path=prod_model_file_path)
            
            if prod_model_object is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=True, \
                    evaluated_model_path=trained_model_file_path
                )
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact
            
            model_list = [trained_model_object,prod_model_object]
            
            # get train & test data (X,y)
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema_content = read_yaml_file(file_path=schema_file_path)
            
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            train_df = load_data(file_path=train_file_path,schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path,schema_file_path=schema_file_path)
            
            target_column = schema_content[TARGET_COLUMN_KEY]
            
            # target column
            logging.info(f"Converting target column into numpy array.")
            target_train_arr = np.array(train_df[target_column]) 
            target_test_arr = np.array(test_df[target_column])
            logging.info(f"Conversion completed target column into numpy array.")
            
            # input feature columns
            logging.info(f"Dropping target column from the dataframe.")
            train_df.drop([target_column],axis=1,inplace=True)
            test_df.drop([target_column],axis=1,inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")
            
            # compare the models:
            metric_info:MetricInfoArtifact = evaluate_regression_model(X_train=train_df,\
                y_train=target_train_arr,X_test=test_df,y_test=target_test_arr,
                base_accuracy=self.model_trainer_artifact.model_accuracy,tol=0.25)
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info}")
            
            # Evaluate the result
            if metric_info is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,\
                    evaluated_model_path=None
                    )
                logging.info(response)
                return response

            if metric_info.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=True,\
                    evaluated_model_path=trained_model_file_path
                    )
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is not better than existing model hence not accepting"+\
                    " trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=False,\
                    evaluated_model_path=prod_model_file_path
                    )
            return model_evaluation_artifact
            
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")