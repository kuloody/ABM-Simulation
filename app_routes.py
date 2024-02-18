import pandas as pd
import numpy as np
import time
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from flask import send_file
from model import SimClass
import os

# Get the directory of the current script
project_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)


# Initialize SimClass instance with initial data
initial_data = pd.read_csv('OldPIPS-SAMPLE.csv')
sim_instance = SimClass(initial_data)

def generateDataset(input1_value,input2_value):
    num_highly = int(input1_value)

    num_moderately = int(input2_value)

    # Read the content of 'dataset.csv' and put it in a dataframe
    df = pd.read_csv('dataset.csv', low_memory=False)

    # Replace '#NULL!' with NaN
    df = df.replace('#NULL!', np.nan)

    # Remove rows with NULL values
    df = df.dropna()

    # Convert relevant columns to numeric types
    numeric_cols = ['Inattentiveness', 'Hyperactivity', 'Impulsiveness', 'Class']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Apply conditions for highly disruptive and moderately disruptive rows
    highly_disruptive = df[(df['Inattentiveness'] > 4) | (df['Hyperactivity'] > 3) | (df['Impulsiveness'] > 0)]
    moderately_disruptive = df[(df['Inattentiveness'] >= 3) | (df['Hyperactivity'] > 1)]

    # Create subset for rows that are neither highly nor moderately disruptive
    neither_disruptive = df[~df.index.isin(highly_disruptive.index) & ~df.index.isin(moderately_disruptive.index)]

    # Sample rows from each subset based on input parameters
    highly_subset = highly_disruptive.sample(n=min(num_highly, len(highly_disruptive)))
    moderately_subset = moderately_disruptive.sample(n=min(num_moderately, len(moderately_disruptive)))
    neither_subset = neither_disruptive.sample(n=30 - len(highly_subset) - len(moderately_subset))

    # Concatenate the subsets
    final_subset = pd.concat([highly_subset, moderately_subset, neither_subset])

    # Shuffle the final subset
    final_subset = final_subset.sample(frac=1).reset_index(drop=True)

    # Write the output to a new file named 'generated.csv'
    final_subset.to_csv('generated.csv', index=False)


@app.route('/generate_dataset')
def generate_dataset():
    try:
        # Get the values of the input fields from the query parameters
        input1_value = request.args.get('input1')
        input2_value = request.args.get('input2')

        # Call the function to generate the dataset with the input values
        generateDataset(input1_value, input2_value)

        # Trigger a popup alert indicating success using JavaScript
        alert_script = "<script>alert('Dataset generated successfully!');</script>"
        df = pd.read_csv('generated.csv')
        # Update the data attribute of the SimClass instance
        sim_instance.data = df
        return alert_script
    except Exception as e:
        # Log the exception
        print(f"An error occurred: {str(e)}")
        # Return a JSON response to the client indicating failure
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    json_data = request.json  # Parse JSON data
    df = pd.DataFrame(json_data)  # Convert JSON data to DataFrame

    # Update the data attribute of the SimClass instance
    sim_instance.data = df

    # Process the DataFrame as needed
    print('Received DataFrame:', df)

    return 'DataFrame received successfully', 200


# Define a route to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/show_results')
def show_results():
    try:
        # Return the HTML file to the client
        # Concatenate the project directory with the file path
        file_path = os.path.join(project_dir, 'results_visualizing.html')

        # Return the HTML file to the client
        return send_file(file_path)
    except Exception as e:
        # Log the exception
        print(f"An error occurred: {str(e)}")
        # Return a JSON response to the client indicating failure
        return jsonify({'success': False, 'error': str(e)}), 500
