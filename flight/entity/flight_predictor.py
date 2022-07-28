from flight.exception import FlightException
from flight.util.util import load_object
from flight.constant import *

import os,sys
import pandas as pd
import numpy as np


class FlightData:
    def __init__(self,Date_of_Journey:np.datetime64,Dep_Time:np.datetime64,Arrival_Time:np.datetime64,Duration:str,\
        Airline:str,Source:str,Destination:str,Total_Stops:str,Price:float=None):
        try:
            self.Date_of_Journey = Date_of_Journey
            self.Dep_Time = Dep_Time
            self.Arrival_Time = Arrival_Time
            self.Duration = Duration
            self.Airline = Airline
            self.Source = Source
            self.Destination = Destination
            self.Total_Stops = Total_Stops
            self.Price = Price
            
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def get_flight_input_data_frame(self):
        try:
            flight_input_dict = self.get_flight_data_as_dict()
            return pd.DataFrame(flight_input_dict)
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def get_flight_data_as_dict(self):
        try:
            input_data = {
                DATE_OF_JOURNEY_KEY : self.Date_of_Journey,
                DEP_TIME_KEY : self.Dep_Time,
                ARRIVAL_TIME_KEY : self.Arrival_Time,
                DURATION_KEY : self.Duration,
                AIRLINE_KEY : self.Airline,
                SOURCE_KEY : self.Source,
                DESTINATION_KEY : self.Destination,
                TOTAL_STOPS_KEY : self.Total_Stops
            }
            
            return input_data
        except Exception as e:
            raise FlightException(e,sys) from e
        

class FlightPredictor:
    def __init__(self,model_dir:str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def get_model_path(self):
        try:
            folder_name = max(list(map(int , os.listdir(self.model_dir))))
            latest_model_dir = os.path.join(self.model_dir,f"{folder_name}")
            model_file_name = os.listdir(latest_model_dir)[0]
            model_file_path = os.path.join(latest_model_dir,model_file_name)
            
            return model_file_path
        except Exception as e:
            raise FlightException(e,sys) from e
        
    def predict(self,X):
        try:
            model_file_path = self.get_model_path()
            model = load_object(file_path=model_file_path)
            flight_price = model.predict(X)
            
            return flight_price
        except Exception as e:
            raise FlightException(e,sys) from e