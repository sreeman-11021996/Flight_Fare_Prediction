from collections import namedtuple


DataIngestionArtifact = namedtuple("DataIngestionArtifact",
[ "train_file_path", "test_file_path", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
["schema_file_path", "report_file_path", "report_page_file_path", "is_validated", "message"])

# we can have a report mentioning the results of all validation to be dropped or
# imputed in Transformation Phase
# Then we can have it detect duplicates,missing_values,outliers

DataTransformationArtifact = namedtuple("DataTransformationArtifact",[
    "transformed_train_file_path","transformed_test_file_path","preprocessed_object_file_path",
    "is_transformed","message"])

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", [
    "is_trained", "message", "trained_model_file_path","train_rmse", "test_rmse", "train_accuracy",
    "test_accuracy","model_accuracy"])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact", [
    "is_model_accepted", "evaluated_model_path"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pushed", "export_model_file_path"])

