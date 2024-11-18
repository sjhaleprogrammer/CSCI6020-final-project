import pandas as pd
import json

def normalize_election_result(result):
    if result in ["Republican", "R"]:
        return "R"
    elif result in ["Democratic", "D", "Democrat"]:
        return "D"
    return result

def excel_to_json(excel_file, json_file):
    # Read the Excel file
    xls = pd.ExcelFile(excel_file)
    
    # List of sheet names (years) to process
    years_to_process = ['2024', '2020', '2016', '2012', '2008', '2004', '2000']
    
    # Initialize the dictionary to hold the data
    data = {}
    
    # Process each specified sheet in the Excel file
    for sheet_name in years_to_process:
        if sheet_name in xls.sheet_names:
            # Read the sheet into a DataFrame
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Replace NaN values with empty strings
            df = df.fillna('')
            
            # Process each row in the DataFrame
            for _, row in df.iterrows():
                state = row['State']
                year_data = row.to_dict()
                year_data['Year'] = sheet_name  # Add the year to the data
                
                # Normalize the election result
                if 'Election Result' in year_data:
                    year_data['Election Result'] = normalize_election_result(year_data['Election Result'])
                
                # Initialize the state data if not already present
                if state not in data:
                    data[state] = {}
                
                # Add the year data to the state data
                data[state][sheet_name] = year_data
    
    # Write the data dictionary to a JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
excel_file = 'VoterResults.xlsx'
json_file = 'VoterResults.json'
excel_to_json(excel_file, json_file)
print("JSON file created successfully.")