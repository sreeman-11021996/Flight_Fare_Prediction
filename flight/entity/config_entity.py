from collections import namedtuple

# all the config info is provided through .yaml file

# ..............................................................................

DataIngestionConfig = namedtuple("DataIngestionConfig",[
    "dataset_download_url","tgz_download_dir","raw_data_dir","ingested_train_dir",
    "ingested_test_dir"])

# dataet_download_url ->  Download url for data
# tgz_download_dir -> Download folder (Compressed file)
# raw_data_dir -> Extract folder (Extracted file)
# ingested_train_dir -> Train Dataset Folder
# ingested_testdir -> Test Dataset Folder

# ..............................................................................

DataValidationConfig = namedtuple("DataValidationConfig",["schema_file_path",
    "report_file_path","report_page_file_path"])

# report_file_path -> include data-drift and so on ->json file
# report_page_file_path -> report.html


# ..............................................................................

DataTransformationConfig = namedtuple("DataTransformationConfig",[
    "transformed_train_dir","transformed_test_dir","preprocessed_object_file_path"])

# transformed_train_dir -> train_trf.csv
# transformed_test_dir  -> test_trf.csv
# preprocessed_oject_file_path -> trf.pkl

# ..............................................................................

ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_model_file_path",
    "base_accuracy","model_config_file_path"])

# base_accuracy -> expectation
# trained_model_file_path -> model.pkl
# model_config_file_path -> model.yaml

# ..............................................................................

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", [
    "model_evaluation_file_path","time_stamp"])

# model_evaluation_file_path -> model_evaluation.yaml
# time_stamp -> curr. time when writing in model_evaluation.yaml

# ..............................................................................

ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])
# export_dir_path -> where to export our new model

# ..............................................................................

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])
# artifact dir -> where all pipeline component results are stored for all cycles