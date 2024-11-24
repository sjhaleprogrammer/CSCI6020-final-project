from flask import Flask, render_template, jsonify, request
import sys
import os
from joblib import load
import pandas as pd
import torch

# Add the parent directory of 'python/' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from python.inference import predict_election_from_state

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('temp_index.html')  # Use the temporary template

@app.route('/predict', methods=['POST'])
def predict_election():
    """
    Predict the election result for a given state and year using the specified model.
    """
    try:
        # Parse input JSON
        data = request.json
        model_name = data.get('model_name')
        state = data.get('state')
        year = data.get('year', 2024)

        if not model_name or not state:
            return jsonify({"error": "model_name and state are required"}), 400

        # Call the core prediction function from inference.py
        result = predict_election_from_state(model_name, state, year)

        return jsonify({
            "state": state,
            "year": year,
            "model": model_name,
            "predicted_result": result
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

