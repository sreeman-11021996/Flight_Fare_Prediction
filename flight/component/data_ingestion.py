from flight.entity.config_entity import DataIngestionConfig
import sys,os
from flight.exception import FlightException
from flight.logger import logging
from flight.entity.artifact_entity import DataIngestionArtifact
import tarfile  # to extract tdz file
import numpy as np
#from six.moves import urllib    # to download url
from urllib import request # to extract tdz file
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:
    
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20} ")
            self.data_ingestion_config = data_ingestion_config
            
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def download_housing_data(self,) -> str:
        try:
            #extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            #folder location to download file
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            
            # clear the folder if it already exists 
            # housing\\artifact\\data_ingestion\\<time_stamp>\\tgz_data -> folder
            if os.path.exists(tgz_download_dir):
                os.remove(tgz_download_dir)
            
            # make a new folder if not exists
            os.makedirs(tgz_download_dir,exist_ok=True)

            # get the file name from url
            housing_file_name = os.path.basename(download_url)

            # make the file path
            tgz_file_path = os.path.join(tgz_download_dir, housing_file_name)
            
            # extract data into the file path
            logging.info(f"Downloading file from :[{download_url}] into :[{tgz_file_path}]")
            request.urlretrieve(download_url, tgz_file_path)
            logging.info(f"File :[{tgz_file_path}] has been downloaded successfully.")
            return tgz_file_path
        
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def extract_tgz_file(self,tgz_file_path:str):
        try:
            # folder location to extract the file
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            # clear the folder if it already exists 
            # housing\\artifact\\data_ingestion\\<time_stamp>\\raw_data -> folder
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
                
            # create the folder -> raw_data
            os.makedirs(raw_data_dir,exist_ok=True)
            
            # Extract the raw_data from tdz_data
            logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir: [{raw_data_dir}]")
            with tarfile.open(tgz_file_path) as housing_tgz_file_obj:
                housing_tgz_file_obj.extractall(path=raw_data_dir)
            logging.info(f"Extraction completed")
            
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def drop_rows_with_exception(self,X : pd.DataFrame, tol = 1, columns = ["Airline","Total_Stops"],)->None:
        try:
            drop_exp = False
            for col in columns:
                col_count = X[col].value_counts()
                exception_cat = col_count[col_count == tol].index.values[0]
                X.drop(X[X[col] == exception_cat].index,axis=0,inplace=True,)
                X.reset_index(drop=True,inplace=True)
                drop_exp = True
                logging.info(f"Column <{col}> : Has category '{exception_cat}' with {tol} row(s)")
           
            if drop_exp:    
                logging.info(f"Exception rows have been Dropped")
        
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def drop_duplicates(self,X : pd.DataFrame):
        try:
            if X.duplicated().sum():
                X.drop_duplicates(ignore_index=True,inplace=True)
                logging.info(f"Duplicate data has been removed from DataFrame")
        
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            # get the path to .csv file
            flight_file_path = os.path.join(raw_data_dir,file_name)

            logging.info(f"Reading csv file: [{flight_file_path}]")
            flight_data_frame = pd.read_csv(flight_file_path)

            # drop rows with only one category count in any columnm
            self.drop_rows_with_exception(X=flight_data_frame)
            # drop Duplicate data
            self.drop_duplicates(X=flight_data_frame)
            
            # Stratify split 
            
            min_price = flight_data_frame["Price"].describe()["min"]
            q_one_price = flight_data_frame["Price"].describe()["25%"]
            q_two_price = flight_data_frame["Price"].describe()["50%"]
            q_three_price = flight_data_frame["Price"].describe()["75%"]
            
            flight_data_frame["Price_cat"] = pd.cut(
                flight_data_frame["Price"],
                bins=[min_price-1, q_one_price, q_two_price, q_three_price, np.inf],
                labels=[1,2,3,4]
            )
            
            
            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index,test_index in split.split(flight_data_frame, flight_data_frame["Price_cat"]):
                strat_train_set = flight_data_frame.loc[train_index].drop(["Price_cat"],axis=1)
                strat_test_set = flight_data_frame.loc[test_index].drop(["Price_cat"],axis=1)
            
            # get the file_paths for train & test
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,file_name)
            
            # Export train & test to their respective folders
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,
                    exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)
            

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                is_ingested=True,
                message=f"Data ingestion completed successfully."
                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact
            
        except Exception as e:
            raise FlightException(e,sys) from e
    
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            tgz_file_path = self.download_housing_data()
            self.extract_tgz_file(tgz_file_path=tgz_file_path)
            data_ingestion_artifact = self.split_data_as_train_test()
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")
         