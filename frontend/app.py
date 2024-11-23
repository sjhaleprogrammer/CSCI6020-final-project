from flask import Flask, render_template
import sys
import os
import sklearn

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the module
from python.inference import *


app = Flask(__name__)

@app.route("/")
def hello_world():

    test = predict_election_from_state("logistic_regression","North Carolina", year=2024)
    
    
    
    return render_template('index.html', output = test)
