from flask import Flask, render_template, jsonify, request, redirect, url_for
import sys
import os
import pandas as pd
import torch

# Add the parent directory of 'python/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from python.inference import predict_election_from_state_with_metrics, predict_election_for_all_states_with_metrics

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/about')
def about():
    return render_template('about.html') 


# Dynamically resolve the file path for electoral_votes.csv
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Directory of app.py
ELECTORAL_VOTES_PATH = os.path.join(BASE_DIR, '..', 'data_raw', 'electoral_votes.csv')
electoral_votes = pd.read_csv(ELECTORAL_VOTES_PATH)

@app.route('/electoral_votes', methods=['GET'])
def get_electoral_votes():
    """
    Serve the electoral votes data as JSON.
    """
    return electoral_votes.set_index('State').to_json()

@app.route('/predict', methods=['POST'])
def predict_election():
    """
    Predict the election result for a single state and year using the specified model.
    """
    try:
        data = request.json
        model_name = data.get('model_name')
        state = data.get('state')
        year = data.get('year', 2024)

        if not model_name or not state:
            return jsonify({"error": "model_name and state are required"}), 400

        if state == "All States":
            # Redirect to /predict_all route
            return redirect(url_for('predict_all_states'), code=307)  # Preserves POST method

        # Call the core prediction function from inference.py
        result = predict_election_from_state_with_metrics(model_name, state, year)

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    
@app.route('/predict_all', methods=['POST'])
def predict_all_states():
    """
    Predict election results for all states for a given year using the specified model.
    """
    try:
        # Parse input JSON
        data = request.json
        model_name = data.get('model_name')
        year = data.get('year', 2024)

        if not model_name:
            return jsonify({"error": "model_name is required"}), 400

        # Call the core function to predict all states
        results = predict_election_for_all_states_with_metrics(model_name, year)

        return jsonify(results)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

