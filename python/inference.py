import os
import pandas as pd
from joblib import load

# Paths
model_path = '../models/logistic_regression/logistic_regression_model.joblib'
preprocessor_path = '../models/logistic_regression/preprocessor.joblib'
dataset_path = '../data_processed/all_states/all_states_prediction.csv'

# Required Features
required_features = [
    'State', 'Electoral College Votes', 'Nationwide Inflation (%)',
    'In Recession (Y/N)', 'Population', 'Unemployment Rate (%)',
    'Median Household Income', '% with Bachelor\'s Degree or Higher',
    '% Without Healthcare Coverage', 'Year'
]

def load_model_and_preprocessor():
    """
    Load the trained model and preprocessor from their respective paths.
    """
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

def predict_election_from_state(state, year=2024):
    """
    Predict the election result for a given state and year.
    
    Args:
        state (str): The state name.
        year (int): The election year.
    
    Returns:
        str: Predicted result ('Republican' or 'Democratic').
    """
    state_data = get_state_data(state, year)
    model, preprocessor = load_model_and_preprocessor()
    
    # Extract features for prediction
    input_data = {feature: state_data[feature] for feature in required_features}
    input_df = pd.DataFrame([input_data])
    
    # Preprocess and predict
    X_transformed = preprocessor.transform(input_df)
    prediction = model.predict(X_transformed)
    
    return 'Republican' if prediction[0] == 0 else 'Democratic'

# Example usage
if __name__ == "__main__":
    try:
        state = "California"
        result = predict_election_from_state(state)
        print(f"Predicted election result for {state}: {result}")
    except Exception as e:
        print(f"Error: {e}")
