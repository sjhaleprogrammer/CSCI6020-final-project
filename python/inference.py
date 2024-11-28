import os
import json
import pandas as pd
import torch
import torch.nn as nn
from joblib import load

# Calculate the absolute base directory of the project
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

# Update paths to be absolute
models_base_path = os.path.join(base_dir, "models/")
dataset_path = os.path.join(base_dir, "data_processed/all_states/all_states_prediction.csv")

# Required Features
required_features = [
    'State', 'Electoral College Votes', 'Nationwide Inflation (%)',
    'In Recession (Y/N)', 'Population', 'Unemployment Rate (%)',
    'Median Household Income', '% with Bachelor\'s Degree or Higher',
    '% Without Healthcare Coverage', 'Year'
]

# Define the neural network
class ElectionPredictor(nn.Module):
    def __init__(self, input_size):
        super(ElectionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # First hidden layer with 32 neurons
        self.fc2 = nn.Linear(32, 16)         # Second hidden layer with 16 neurons
        self.fc3 = nn.Linear(16, 1)          # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

def load_model_and_preprocessor(model_name):
    """
    Load the trained model and preprocessor for the specified model.
    
    Args:
        model_name (str): The name of the model folder (e.g., "logistic_regression").
    
    Returns:
        model, preprocessor: Loaded model and preprocessor.
    """
    model_folder_path = os.path.join(models_base_path, model_name)
    preprocessor_path = os.path.join(model_folder_path, "preprocessor.joblib")
    
    # Handle different file extensions for models
    if model_name == "neural_network":
        model_path = os.path.join(model_folder_path, f"{model_name}_model.pth")
    else:
        model_path = os.path.join(model_folder_path, f"{model_name}_model.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
    
    # Load the preprocessor
    preprocessor = load(preprocessor_path)
    
    # Load the model
    if model_name == "neural_network":
        input_size = len(preprocessor.transformers_[0][2]) + len(
            preprocessor.transformers_[1][1].get_feature_names_out()
        )
        model = ElectionPredictor(input_size)  # Initialize the neural network structure
        model.load_state_dict(torch.load(model_path))  # Load trained weights
        model.eval()  # Set the model to evaluation mode
    else:
        model = load(model_path)
    
    return model, preprocessor

def get_state_data(state, year, dataset_path=dataset_path):
    """
    Retrieve the row of data for a given state and year from the dataset.
    
    Args:
        state (str): The state name.
        year (int): The year to query.
        dataset_path (str): Path to the dataset CSV.
    
    Returns:
        dict: A dictionary of features for the given state and year.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    data = pd.read_csv(dataset_path)
    row = data[(data['State'] == state) & (data['Year'] == year)]
    if row.empty:
        raise ValueError(f"No data found for state: {state} in year: {year}")
    return row.iloc[0].to_dict()

def load_metrics(model_name):
    """
    Load the evaluation metrics for the specified model.
    
    Args:
        model_name (str): The name of the model folder (e.g., "logistic_regression").
    
    Returns:
        dict: A dictionary containing the model's metrics.
    """
    metrics_path = os.path.join(models_base_path, model_name, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    return metrics

def predict_election_from_state_with_metrics(model_name, state, year=2024):
    """
    Predict the election result for a given state and year using the specified model,
    and return the prediction along with evaluation metrics.
    
    Args:
        model_name (str): The name of the model to use (e.g., "logistic_regression").
        state (str): The state name.
        year (int): The election year.
    
    Returns:
        dict: A dictionary containing the predicted result and model metrics.
    """
    state_data = get_state_data(state, year)
    model, preprocessor = load_model_and_preprocessor(model_name)
    metrics = load_metrics(model_name)
    
    # Extract features for prediction
    input_data = {feature: state_data[feature] for feature in required_features}
    input_df = pd.DataFrame([input_data])
    
    # Preprocess and predict
    X_transformed = preprocessor.transform(input_df)
    if model_name == "neural_network":
        X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(X_tensor).item()
        predicted_result = 'Democratic' if prediction > 0.5 else 'Republican'
    else:
        prediction = model.predict(X_transformed)
        predicted_result = 'Democratic' if prediction[0] == 1 else 'Republican'
    
    # Combine prediction with metrics
    return {
        "state": state,
        "year": year,
        "model": model_name,
        "predicted_result": predicted_result,
        "metrics": metrics
    }

def predict_election_for_all_states_with_metrics(model_name, year=2024):
    """
    Predict election results for all states for a given year using the specified model,
    and include model metrics in the response.

    Args:
        model_name (str): The name of the model to use (e.g., "logistic_regression").
        year (int): The election year.

    Returns:
        dict: A dictionary containing predictions for all states and model metrics.
    """
    # Load model, preprocessor, and metrics
    model, preprocessor = load_model_and_preprocessor(model_name)
    metrics = load_metrics(model_name)

    # Load the dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    data = pd.read_csv(dataset_path)

    # Get unique states
    all_states = data['State'].unique()
    predictions = {}

    # Loop through each state
    for state in all_states:
        try:
            # Get state data
            state_data = get_state_data(state, year)
            input_data = {feature: state_data[feature] for feature in required_features}
            input_df = pd.DataFrame([input_data])

            # Preprocess data
            X_transformed = preprocessor.transform(input_df)

            # Predict
            if model_name == "neural_network":
                X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
                with torch.no_grad():
                    prediction = model(X_tensor).item()
                predicted_result = "Democratic" if prediction > 0.5 else "Republican"
            else:
                prediction = model.predict(X_transformed)
                predicted_result = "Democratic" if prediction[0] == 1 else "Republican"

            # Store result
            predictions[state] = {
                "predicted_result": predicted_result,
                "metrics": metrics
            }

        except Exception as e:
            # Handle errors for individual states
            predictions[state] = {"error": str(e)}

    # Combine predictions with model metrics
    return {
        "predictions": predictions,
        "metrics": metrics
    }