from flask import Flask, render_template, jsonify, request
import sys
import os
import sklearn



# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from python.inference import *


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 


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
        year = data.get('year',2024)
        
        if not model_name or not state:
            return jsonify({"error": "model_name and state are required"}), 400
            
            
        
        dataset_path = 'data_processed/all_states/all_states_prediction.csv'   
            
        model_path = os.path.join("models/", model_name, f"{model_name}_model.joblib")
        preprocessor_path = os.path.join("models/", model_name, "preprocessor.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        
        model = load(model_path)
        preprocessor = load(preprocessor_path)
       

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
        data = pd.read_csv(dataset_path)
        row = data[(data['State'].str.lower() == state.lower()) & (data['Year'] == year)]
        
        if row.empty:
            raise ValueError(f"No data found for state: {state} in year: {year}")
        state_data = row.iloc[0].to_dict()
        
        # Extract features for prediction
        input_data = {feature: state_data[feature] for feature in required_features}
        input_df = pd.DataFrame([input_data])
        
        # Preprocess and predict
        X_transformed = preprocessor.transform(input_df)
        prediction = model.predict(X_transformed)
        
        result = 'Republican' if prediction[0] == 0 else 'Democratic'
        
        
        return jsonify({
            "state": state,
            "year": year,
            "predicted_result": result
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500



