from flask import Flask, request
import sys

import pip
from flight.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from flight.logger import logging,get_log_dataframe
from flight.exception import FlightException
import os, sys
import json
from flight.config.configuration import Configuration
from flight.constant import CONFIG_DIR, get_current_time_stamp
from flight.pipeline.pipeline import Pipeline
from flight.entity.flight_predictor import FlightPredictor, FlightData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "flight"
SAVED_MODELS_DIR_NAME = "saved_models"

MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


FLIGHT_DATA_KEY = "flight_data"
FLIGHT_PRICE_KEY = "Flight Price"

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.experiment.get_experiment_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": Pipeline.experiment.get_experiment_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        FLIGHT_DATA_KEY: None,
        FLIGHT_PRICE_KEY: None
    }
    if request.method == 'POST':
        Airline = str(request.form['Airline'])
        Date_of_Journey = request.form['Date_of_Journey']
        Source = str(request.form['Source'])
        Destination = str(request.form['Destination'])
        Dep_Time = request.form['Dep_Time']
        Arrival_Time = request.form['Arrival_Time']
        Duration = str(request.form['Duration'])
        Total_Stops = str(request.form['Total_Stops'])
        
        flight_data = FlightData(
            Airline=Airline,
            Date_of_Journey=Date_of_Journey,
            Source=Source,
            Destination=Destination,
            Dep_Time=Dep_Time,
            Arrival_Time=Arrival_Time,
            Duration=Duration,
            Total_Stops=Total_Stops                     
            )
        
        flight_df = flight_data.get_flight_input_data_frame()
        flight_predictor = FlightPredictor(model_dir=MODEL_DIR)
        flight_price = flight_predictor.predict(X=flight_df)
        context = {
            FLIGHT_DATA_KEY: flight_data.get_flight_data_as_dict(),
            FLIGHT_PRICE_KEY: flight_price,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    context = None
    if request.method == "POST":
        test_file = request.files['Upload File']
        print(test_file)
        return render_template("batch_predict.html",context=context)
    return render_template("batch_predict.html", context=context)



if __name__ == "__main__":
    app.run(port=5000)
    
    