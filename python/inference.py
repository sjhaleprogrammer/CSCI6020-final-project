import os
import pandas as pd
from joblib import load

# Base paths
models_base_path = '../models/'
dataset_path = '../data_processed/all_states/all_states_prediction.csv'

# Required Features
required_features = [
    'State', 'Electoral College Votes', 'Nationwide Inflation (%)',
    'In Recession (Y/N)', 'Population', 'Unemployment Rate (%)',
    'Median Household Income', '% with Bachelor\'s Degree or Higher',
    '% Without Healthcare Coverage', 'Year'
]

def load_model_and_preprocessor(model_name):
    """
    Load the trained model and preprocessor for the specified model.
    
    Args:
        model_name (str): The name of the model folder (e.g., "logistic_regression").
    
    Returns:
        model, preprocessor: Loaded model and preprocessor.
    """
    model_path = os.path.join(models_base_path, model_name, f"{model_name}_model.joblib")
    preprocessor_path = os.path.join(models_base_path, model_name, "preprocessor.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
    
    model = load(model_path)
    preprocessor = load(preprocessor_path)
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

def predict_election_from_state(model_name, state, year=2024):
    """
    Predict the election result for a given state and year using the specified model.
    
    Args:
        model_name (str): The name of the model to use (e.g., "logistic_regression").
        state (str): The state name.
        year (int): The election year.
    
    Returns:
        str: Predicted result ('Republican' or 'Democratic').
    """
    state_data = get_state_data(state, year)
    model, preprocessor = load_model_and_preprocessor(model_name)
    
    # Extract features for prediction
    input_data = {feature: state_data[feature] for feature in required_features}
    input_df = pd.DataFrame([input_data])
    
    # Preprocess and predict
    X_transformed = preprocessor.transform(input_df)
    prediction = model.predict(X_transformed)
    
    return 'Republican' if prediction[0] == 0 else 'Democratic'

