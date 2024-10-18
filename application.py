import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle 

from flask import Flask, render_template, request, jsonify

application = Flask(__name__)
app=application

scaler = pickle.load(open('Models/scaler.pkl', 'rb'))
linear_model = pickle.load(open('Models/linear_model.pkl', 'rb'))

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=["POST","GET"])
def predict_fires():
    if request.method == "POST":

        result = 0
        
        Temperature = float(request.form.get('temperature'))
        RH = float(request.form.get('rh'))
        WS = float(request.form.get('ws'))
        Rain = float(request.form.get('rain'))
        FFMC = float(request.form.get('ffmc'))
        DMC = float(request.form.get('dmc'))
        ISI = float(request.form.get('isi'))
        Classes = float(request.form.get('classes'))
        Region = float(request.form.get('region'))

        form_data_scaled = scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])

        prediction_output  = abs(linear_model.predict(form_data_scaled))

        result = round(prediction_output[0] ,2 )


        return render_template('home.html', results=result)


    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

